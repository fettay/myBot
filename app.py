import handler
import os
import sys
import json
from flask import Flask, request
import pandas as pd
from sessionHandler import SessionHandler
import redis
import fbApi


app = Flask(__name__)

DATA_LOC = 'Data/'
PRODUCTS = pd.read_csv(DATA_LOC + 'Product.csv').fillna('')
SHOPS = pd.read_csv(DATA_LOC + 'Shops2.csv').fillna('')
hdl = handler.Handler(opt_list=handler.ALL_OPT, shops=SHOPS, products=PRODUCTS)

# Test setting
TEST_MODE = True if len(sys.argv) > 1 and sys.argv[1] == 'test' else False

if TEST_MODE:
    FB_URL = "http://localhost:9999/"
    ACCESS_TOKEN = ""
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
else:
    FB_URL = "https://graph.facebook.com/v2.6/me/"
    ACCESS_TOKEN = os.environ["PAGE_ACCESS_TOKEN"]
    REDIS_URL = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
    r = redis.from_url(REDIS_URL)

PARAMS = {
    "access_token": ACCESS_TOKEN
}
HEADERS = {
    "Content-Type": "application/json"
}

sess_handler = SessionHandler(r)


@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must
    # return the 'hub.challenge' value in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == os.environ["VERIFY_TOKEN"]:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200

    return "Hello world", 200


@app.route('/', methods=['POST'])
def webook():
    # endpoint for processing incoming messaging events
    data = request.get_json()
    print(data)  # you may not want to log every incoming message in production, but it's good for testing

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:
                if messaging_event.get("message"):  # someone sent us a message
                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text
                    # Get Session
                    current_sess = sess_handler.get(sender_id)
                    # Handle response: De quelle ... parlez vous?
                    # if 'classify' in current_sess and current_sess['classify'][1] == 1:
                    #     cls_result = hdl.classify(message_text, class_=current_sess['classify'][0])
                    # else:
                    cls_result = hdl.classify(message_text)

                    if cls_result[0] is not None:
                        current_sess.set(classify=cls_result)
                    responses_message = hdl.responses_formatter(cls_result, message_text)
                    action_fn = globals()[responses_message[0]]  # Requested action
                    action_fn(sender_id, responses_message[1])
                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass

    return "ok", 200


def send_message(recipient_id, message_text):

    print("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    data = {
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    }
    fbApi.message(params=PARAMS, headers=HEADERS, data=data)


def send_carousel(recipient_id, formatted_carousel):
    print("sending carousel to {recipient}".format(recipient=recipient_id))

    formatted_carousel["recipient"] = {"id": recipient_id}
    data = formatted_carousel
    fbApi.message(params=PARAMS, headers=HEADERS, data=data)


@app.errorhandler(500)
def internal_error(error):
    print("ERROR:" + str(error))
    return "500 error"

if __name__ == '__main__':
    DEFAULT_GREETING = "Bonjour, je peux vous indiquez les prix des articles, ou les horaires des magasins."
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    # fbApi.set_greetings(DEFAULT_GREETING)


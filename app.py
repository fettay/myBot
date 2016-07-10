import handler
import os
import sys
import json
import requests
from flask import Flask, request
import pandas as pd
from sessionHandler import SessionHandler
from session import Session

app = Flask(__name__)

DATA_LOC = 'Data/'
ALL_OPT = ['product_price', 'shop_hours', 'shop_location', 'shop_telephone']
PRODUCTS = pd.read_csv(DATA_LOC + 'Product.csv')
SHOPS = pd.read_csv(DATA_LOC + 'Shops.csv')
hdl = handler.Handler(opt_list=ALL_OPT, shops=SHOPS, products=PRODUCTS)

# Test setting
TEST_MODE = True if len(sys.argv) > 1 and sys.argv[1] == 'test' else False
if TEST_MODE:
    FB_URL = "http://localhost:9999/"
    ACCESS_TOKEN = ""
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
else:
    FB_URL = "https://graph.facebook.com/v2.6/me/"
    ACCESS_TOKEN = os.environ["PAGE_ACCESS_TOKEN"]
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379


PARAMS = {
    "access_token": ACCESS_TOKEN
}
HEADERS = {
    "Content-Type": "application/json"
}

sess_handler = SessionHandler(host=REDIS_HOST, port=REDIS_PORT)
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
    log(data)  # you may not want to log every incoming message in production, but it's good for testing

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:
                if messaging_event.get("message"):  # someone sent us a message
                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text
                    # Get Session
                    current_sess = sess_handler.get(sender_id)
                    if 'classify' in current_sess:
                        print("Was a %s" % current_sess["classify"])
                    cls_result = hdl.classify(message_text)
                    if cls_result is not None:
                        current_sess.set(classify=cls_result)
                    responses_message = hdl.responses_formatter(cls_result, message_text)
                    send_message(sender_id, responses_message)
                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass

    return "ok", 200


def send_message(recipient_id, message_text):

    log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    })
    r = requests.post(FB_URL + "messages", params=PARAMS, headers=HEADERS, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)


def set_greetings(greeting):
    data = {"setting_type": "greeting",
            "greeting": {
                    "text": greeting
                }
            }
    r = requests.post(FB_URL + "thread_settings", params=PARAMS, headers=HEADERS, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)


def log(message):  # simple wrapper for logging to stdout on heroku
    print(str(message))
    sys.stdout.flush()

@app.errorhandler(500)
def internal_error(error):
    print(error)
    return "500 error"

if __name__ == '__main__':
    DEFAULT_GREETING = "Bonjour, je peux vous indiquez les prix des articles, ou les horaires des magasins."
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    # set_greetings(DEFAULT_GREETING)

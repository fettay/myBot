__author__ = 'raphaelfettaya'

from flask import Flask, request, render_template
import requests
import os
import json

app = Flask(__name__)
TEST_APP_URL = "http://localhost:5000/"
HEADERS = {'content-type': 'application/json'}


@app.route('/')
def sender():
    return render_template('send_message.html')

# TODO DEBUG THAT TO GET A REAL CONV
# @app.route('/messages', methods=['POST'])
# def received():
#     data = request.get_json()
#     print("Received message:")
#     print(data["message"]["text"])
#     print("")
#     return "ok", 200


@app.route('/send', methods=['POST'])
def send():
    data = request.get_json()
    print("Sent message:")
    print(data["text"])
    print("")
    formatted = format_fb_req(data["text"])
    r = requests.post(TEST_APP_URL, data=formatted, headers=HEADERS)
    return "ok", 200


def format_fb_req(text):
    data = {
        "object": "page",
        "entry": [{
                "messaging": [{
                    "message": {
                        "text": text
                    },
                    "sender":{
                        "id": 0
                    },
                    "recipient":{
                        "id": 0
                    }
                }]
            }]
    }
    return json.dumps(data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9999))
    app.run(port=port)
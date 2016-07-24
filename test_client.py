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

@app.route('/send_location', methods=['POST'])
def send_location():
    send_url = 'http://freegeoip.net/json'
    r = requests.get(send_url)
    j = json.loads(r.text)
    lat, long = j['latitude'], j['longitude']
    # lat, long = 48.856614, 2.352222  # Paris HDV
    req = pos_req((lat, long))
    r = requests.post(TEST_APP_URL, data=req, headers=HEADERS)
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


def pos_req(pos):
    data = {'object': 'page',
            'entry': [{'id': '927198464073529', 'time': 1469369553565, 'messaging':
                [{'message': {'seq': 1790, 'attachments':
                    [{'url': 'https://www.facebook.com/l.php?u=https%3A%2F%2Fwww.bing.com%2Fmaps%2Fdefault.aspx%3Fv%3D2%26pc%3DFACEBK%26mid%3D8100%26where1%3D32.074129162848%252C%2B34.791280477752%26FORM%3DFBKPL1%26mkt%3Den-US&h=WAQFZAN8m&s=1&enc=AZMsEiZgzyVw82MP-LJHUsNdfxENGFp5A_0y24oSNMHA-rI8ipCZ1aUBiRjUJNhFGm3mdAXN-sGn9_1KgtYtGxoLTpo8dxfU9LklranAJGvBZA',
                      'title': "Raphaël's Location", 'type': 'location',
                      'payload': {'coordinates': {'lat': pos[0], 'long': pos[1]}}}], 'mid': 'mid.1469369538361:069abbab01e0f71710'},
                  'timestamp': 1469369538920, 'sender': {'id': '0'}, 'recipient': {'id': '0'}}]}]}
    return json.dumps(data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9999))
    app.run(port=port)
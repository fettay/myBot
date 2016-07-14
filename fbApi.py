__author__ = 'raphaelfettaya'
import requests
import sys
import app
import json
import os


def message(params, headers, data):
    # if "RECIPIENT_TEST" in os.environ:
    #     data["recipient"]["id"] == os.environ['RECIPIENT_TEST']
    data = json.dumps(data)
    if app.TEST_MODE:
        print(data)
        return
    r = requests.post(app.FB_URL + "messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)

def set_greetings(greeting):
    data = {"setting_type": "greeting",
            "greeting": {
                    "text": greeting
                }
            }
    data = json.dumps(data)
    r = requests.post(app.FB_URL + "thread_settings", params=app.PARAMS, headers=app.HEADERS, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)



def log(message):  # simple wrapper for logging to stdout on heroku
    print(str(message))
    sys.stdout.flush()
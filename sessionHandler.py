import json
from session import Session
from datetime import datetime
TIME_FORMAT = "%d %m %y %H:%M:%S"


class SessionHandler(object):

    def __init__(self, redis):
        self.redis_db = redis

    def create_session(self, id, **kwargs):
        kwargs['created'] = datetime.now().strftime(TIME_FORMAT)
        json_data = json.dumps(kwargs)
        self.redis_db.set(id, json_data)

    def get(self, id):
        sess_data = self.redis_db.get(id)
        if sess_data is None:
            self.create_session(id)
            return Session(id, self)
        sess_data = sess_data.decode('utf-8')
        sess_data = json.loads(sess_data)
        return Session(id, self, **sess_data)

    def set(self, sess, **kwargs):
        data = sess.data
        self.redis_db.set(sess.id, json.dumps(data))
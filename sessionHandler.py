import redis
import json
from session import Session
from datetime import datetime
TIME_FORMAT = "%d %m %y %H:%M:%S"


class SessionHandler(object):

    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_db = redis.StrictRedis(host=host, port=port, db=db)

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
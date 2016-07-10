class Session(object):

    def __init__(self, id, s_handler, **kwargs):
        self.id = id
        self.handler = s_handler
        self.data = {}
        for k, v in kwargs.items():
            self.data[k] = v

    def __get__(self, k):
        return self.data[k]

    def __set__(self, k, v):
        self.data[k] = v
        to_change = {k: v}
        self.handler.set(self.id, **to_change)

    def __iter__(self):
        for k, v in self.data.items():
            yield k, v

    def set(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k] = v
        self.handler.set(self, **kwargs)
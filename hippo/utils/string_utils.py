import uuid


def get_uuid(numchars=4):
    return uuid.uuid4().hex[:numchars]
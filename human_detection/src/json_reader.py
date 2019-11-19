import json
from pprint import pprint


def read_json(filename):
    with open(filename) as f:
        data = json.load(f)

    # pprint(data)
    return data

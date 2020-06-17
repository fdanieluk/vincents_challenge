# import random
from time import sleep
import json
import requests as rq
from pathlib import Path
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


raw_data = Path('data/raw')


def send_emb(emb):
    url = "http://challenge.calmcode.io/attempt/"
    payload = {"user": "Filip Danieluk",
               "email": "filip.danieluk@cbre.com",
               "emb": emb}
    resp = rq.post(url, json=payload)
    sleep(.01)
    return {**payload, **resp.json()}  # merges payload with response


def get_score(emb):
    data = send_emb(emb)

    file_name = f'{data["datetime"]}_{data["distance"]}.json'.replace(':', '-')
    file_path = raw_data / file_name
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)
    return data


def score(params):
    emb = list(params.values())

    distances = []
    for i in range(0, 100):
        distances.append(get_score(emb)['distance'])
    loss = min(distances)
    print(loss)
    print('\n')
    return {'loss': loss, 'status': STATUS_OK}


def optimize(random_state=23):
    space = {f'{i}': hp.quniform(f'{i}', -5, 5, .01) for i in range(0, 100)}

    best = fmin(score,
                space,
                algo=tpe.suggest,
                max_evals=5000)
    return best


if __name__ == '__main__':
    optimize()

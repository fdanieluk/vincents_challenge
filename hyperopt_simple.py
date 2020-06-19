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

    file_path = raw_data / f'data_dump.json'
    with open(file_path, 'a') as outfile:
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
    # optimize()
    emb = [-4.5720e-01, -9.1640e-01,  3.5332e+00, -3.5431e+00, -2.8393e+00,
        3.7930e-01,  1.7025e+00,  2.5669e+00, -3.9023e+00, -1.8006e+00,
        4.4121e+00,  1.8441e+00,  1.4642e+00,  2.2680e+00,  3.5787e+00,
       -3.4024e+00, -2.3458e+00,  5.0340e-01,  4.5317e+00, -6.9070e-01,
        4.3477e+00, -1.5115e+00, -8.7460e-01, -3.0177e+00, -8.2710e-01,
       -1.6237e+00,  1.3380e+00, -3.8580e+00, -3.6129e+00,  1.7350e+00,
       -2.9706e+00,  3.2682e+00, -1.7649e+00, -2.5511e+00,  3.5552e+00,
       -1.3062e+00,  3.4763e+00, -2.7880e+00,  3.3180e+00, -3.4952e+00,
        1.2583e+00, -3.0097e+00, -3.6837e+00, -1.8222e+00,  4.2390e+00,
       -3.4367e+00, -3.2920e+00, -4.2652e+00,  9.5660e-01, -1.5293e+00,
        2.8594e+00, -1.2000e-03,  4.3955e+00,  2.0996e+00,  9.1400e-01,
       -4.3763e+00,  2.5016e+00, -3.4963e+00, -2.0391e+00,  4.0380e-01,
        4.4068e+00,  2.4314e+00, -5.5390e-01, -2.3599e+00, -3.6408e+00,
       -3.6312e+00, -3.9557e+00,  3.7370e+00,  3.2012e+00,  3.2128e+00,
        1.1688e+00,  3.4348e+00, -1.6696e+00,  4.2402e+00,  3.4357e+00,
        3.4448e+00,  4.1009e+00, -4.6878e+00,  2.5991e+00,  1.6586e+00,
        3.2382e+00, -4.3908e+00,  1.1406e+00,  7.1220e-01,  3.3214e+00,
        2.7498e+00, -4.2622e+00, -1.3719e+00, -1.2032e+00, -2.1542e+00,
        7.6740e-01,  2.5001e+00,  1.6693e+00, -3.2734e+00,  2.2683e+00,
       -4.7540e-01, -6.8700e-02,  1.9135e+00,  9.3680e-01, -1.4474e+00]

    scores = []
    for i in range(500):
        scores.append(get_score(emb)['distance'])
    print(sorted(scores))

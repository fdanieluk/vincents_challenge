from time import sleep, time
import json
import requests as rq
from pathlib import Path
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.optimize import minimize
import numpy as np

s = rq.Session()

raw_data = Path('data/raw')

INITIAL_EMB = [  0.14478454,   0.66783074,   6.12833174,  -2.21519954,
                -4.39568455,  -2.56623048,  -3.28871097,  -0.10629463,
                -5.8746842 ,  -2.67853701,   5.14971444,  -0.69189045,
                 0.2439952 ,   5.45417224,   5.47683426,  -7.12859273,
                -5.61650568,   5.94522712,   2.43361725,  -1.74518626,
                 6.42345499,   1.40225869,  -0.81790141,  -3.23392256,
                -3.77792672,  -2.43647702,   1.53751351,  -6.38037784,
                 2.08120195,  -4.79150766,  -0.94509392,   0.97901918,
                -0.04724608, -11.61999077,   6.28620618,  -3.17255635,
                 2.28672699,  -1.13979271,   6.00342774,  -0.12176718,
                -6.0341162 ,  -4.8943485 , -11.18448442,  -1.61683134,
                 7.15232928,  -6.29149558,   0.64938784,  -8.81651699,
                -3.90449393,   3.10826041,   7.71024944,   3.99594773,
                 4.20523548,  -4.60991234,   2.42025819, -11.01902555,
                -4.16313249,  -0.03538169,  -3.94927192,  -4.56422219,
                10.77098184,   2.11944382,  -1.08229824,   2.05007051,
                -0.4665908 ,  -9.50792733,  -3.89306236,   1.35310912,
                 9.89691343,   3.91600557,   2.02212538,   3.38419418,
               -12.54854751,   9.50303607,  -0.06195468,   1.54059961,
                 4.35444334, -12.54029335,  -2.18132432,   1.71752216,
                 9.52197252,  -4.28708208,  -0.01663795,  -3.90734523,
                 4.13781807,   0.47985612,  -7.0506923 ,  -0.56358526,
                 4.08841836,  -0.73146255,   1.14717618,   6.55668713,
                 0.07206778,   2.9172383 ,  -2.38179778,   5.29257014,
                 2.44454439,  -1.57098411,   0.92158976,  -6.35711194]


def send_emb(emb):
    start = time()
    emb = emb.tolist()
    url = "http://challenge.calmcode.io/attempt/"
    payload = {"user": "Filip Danieluk",
               "email": "filip.danieluk@cbre.com",
               "emb": list(emb)}
    resp = s.post(url, json=payload)
    stop = time()
    # print(stop - start)
    return {**payload, **resp.json()}  # merges payload with response


def get_score(emb):
    data = send_emb(emb)

    file_path = raw_data / f'data_dump.json'
    with open(file_path, 'a') as outfile:
        json.dump(data, outfile)
    return data


def score(params):
    # sort po intach
    # print(params)
    emb = list(params.values())
    distances = []
    for i in range(0, 500):
        distances.append(get_score(emb)['distance'])
    loss = min(distances)
    print(loss)
    print('\n')
    return {'loss': loss, 'status': STATUS_OK}


def optimize(random_state=23):
    space = {f'{i}'.zfill(3): hp.normal(f'{i}', VAL, 1.5) for i, VAL in enumerate(INITIAL_EMB)}

    best = fmin(score,
                space,
                algo=tpe.suggest,
                max_evals=5000)
    return best

def scipy_score(params):
    distances = []
    for i in range(0, 333):
        distances.append(get_score(params)['distance'])
    loss = min(distances)
    print(loss)
    print('\n')
    return loss

if __name__ == '__main__':
    # optimize()
    x0 = np.array(INITIAL_EMB)
    res = minimize(scipy_score, x0, method='powell', options={'disp': True, 'maxiter':1000})
    s.close()

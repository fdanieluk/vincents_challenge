from time import sleep, time
import json
import requests as rq
from pathlib import Path
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.optimize import minimize
import numpy as np

s = rq.Session()

raw_data = Path('data/raw')

INITIAL_EMB = [ -1.74179133,   0.17612164,   8.54870761,   6.03522944,
        -5.1371889 ,  -1.52907496,  -1.86929885,   0.95683249,
        -9.52568015,   2.72657879,   9.85039466,   4.84582731,
        -1.43267092,   6.93672475,   2.79929777,  -4.65851785,
        -7.9945105 ,   5.83823551,   8.2057442 ,   3.20186196,
         6.38333607,  -1.08184216,  -5.34384211,  -4.02546714,
        -2.16426134,  -3.56889002,   6.1196189 ,  -3.73702305,
         0.65658774,  -8.34335541,   1.59297612,   0.71200688,
        -2.15621057,  -6.36091751,   3.21209777,  -2.34359557,
         2.30312576,   1.14181762,   8.18159317,  -2.6155886 ,
        -2.04632924,  -0.78547952,  -8.98433281,  -5.18788751,
         6.97037193,  -6.59183277,  -0.64009988, -10.2547799 ,
        -0.67843336,   5.37166507,   6.37879421,   5.80121363,
         3.87485653,  -4.07522776,   5.76703932, -10.21826549,
        -2.06578925,  -3.9644058 ,  -6.80634523,  -7.66811127,
         8.1489385 ,   7.87592222,   0.90819968,   5.28501885,
        -2.2948264 , -11.0871891 ,  -6.59643872,  -3.22530331,
        10.25544354,   7.34873478,   7.29957203,   1.03155   ,
        -8.89498911,   9.04790227,   0.6907614 ,   0.61702109,
         9.3966957 ,  -9.31877123,  -0.83518522,   5.37276938,
         7.1444983 ,  -2.8465341 ,   2.47982499,  -0.7791078 ,
         8.51264429,   2.96336363,  -0.69265803,   0.907953  ,
         3.58165914,   0.30633438,   5.34269133,   1.33967289,
        -3.47543495,  -1.67490187,   2.48046937,   7.01637129,
         2.14210503,   1.41505156,   1.36150413,  -2.52574848]


def send_emb(emb):
    start = time()
    emb = emb.tolist() # FIXME: check if instance of np.array then do ignore if not
    # print(emb)
    url = "http://challenge.calmcode.io/attempt/"
    payload = {"user": "Filip Danieluk",
               "email": "filip.danieluk@cbre.com",
               "emb": list(emb)}
    resp = s.post(url, json=payload)
    stop = time()
    # print(stop - start)
    # print(resp.json()['distance'])
    return {**payload, **resp.json()}  # merges payload with response


def get_score(emb):
    data = send_emb(emb)

    file_path = raw_data / f'data_dump.json'
    with open(file_path, 'a') as outfile:
        json.dump(data, outfile)
    return data


def score(params):
    # emb = list(params.values())
    distances = []
    for i in range(0, 333):
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
    # for i in range(0, 1): # emb debug
        distances.append(get_score(params)['distance'])
    loss = min(distances)
    print(loss)
    print('\n')
    return loss

if __name__ == '__main__':

    # DO NOT CHANGE THIS IS OFFICIAL!!!


    # optimize()
    x0 = np.array(INITIAL_EMB)
    res = minimize(scipy_score, x0, method='powell', options={'disp': True, 'maxiter':1000})
    # # res = minimize(scipy_score, x0, method='Nelder-Mead', options={'disp': True,
    # #                                                                'maxiter':1000,
    # #                                                                'adaptive': True})
    # emb = [v for v in INITIAL_EMB]
    # for i in range(1):
    #     score(emb)
    s.close()

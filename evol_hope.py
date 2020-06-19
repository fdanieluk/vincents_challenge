import random
from evol import Population, Evolution
from time import sleep, time
import json
import requests as rq
from pathlib import Path
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.optimize import minimize
import numpy as np

s = rq.Session()

raw_data = Path('data/raw')

random.seed(42)

FILE_NAME = f'data_dump_{time().__str__()}.json'

def send_emb(emb):
    start = time()
    if type(emb) != list:
        emb = emb.tolist() # FIXME: check if instance of np.array then do ignore if not
    # print(emb)
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

    file_path = raw_data / FILE_NAME
    with open(file_path, 'a') as outfile:
        json.dump(data, outfile)
    return data


def score(params):
    emb = list(params.values())
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

def random_start():
    INITIAL_EMB = [ -2.92591712,   0.10430707,   7.91638512,   5.77455177,
                    -4.33878822,  -0.14618958,  -2.18440794,   0.59964074,
                    -9.93470577,   3.2856607 ,   8.89438428,   4.73188562,
                    -1.05398829,   6.21786923,   2.66259699,  -4.12366102,
                    -8.56851206,   5.90244994,   8.32513394,   1.85173262,
                     6.63299123,  -0.38769569,  -5.22694104,  -3.64179867,
                    -2.52412888,  -3.81289046,   6.43500955,  -3.45690098,
                     1.08675078,  -8.32170023,   1.66997177,   0.30163379,
                    -1.84988529,  -6.42522649,   3.70392029,  -2.39693782,
                     1.98977755,   1.46874173,   8.25352094,  -2.43083541,
                    -2.91337546,  -0.11664101,  -8.24668035,  -3.96616587,
                     7.11498323,  -6.776439  ,  -0.60890586, -10.52835629,
                    -0.44917007,   5.2864028 ,   6.64425776,   5.08779726,
                     3.88472353,  -4.69346496,   5.99482694,  -9.50208289,
                    -2.74393186,  -4.31508455,  -6.57199462,  -6.96878609,
                     8.06797348,   7.40507447,  -0.0434593 ,   5.26566426,
                    -2.66269295, -10.37857441,  -6.11397387,  -3.23007717,
                     9.73496514,   7.11485276,   6.49493056,   0.07752089,
                    -8.95163408,   9.89474137,   0.91518764,   0.38958969,
                     8.51665069,  -9.676863  ,  -0.61253266,   5.29313682,
                     7.68340523,  -2.57979484,   2.61024503,  -1.35027085,
                     8.37921838,   2.34719994,  -1.28888663,   0.71028515,
                     3.78946895,   0.56683551,   6.34041053,   2.10141822,
                    -3.8342684 ,  -2.8030365 ,   0.99662692,   6.68252198,
                     2.46274858,   0.29371417,   0.32536334,  -3.45471717]
    len(INITIAL_EMB)

    return INITIAL_EMB

def func_to_optimise(emb):
    return scipy_score(emb)

def pick_random_parents(pop):
    mom = random.choice(pop)
    dad = random.choice(pop)
    return mom, dad

def make_child(mom, dad):
    child = (np.array(mom) + np.array(dad))/2
    return child

def add_noise(chromosome, sigma):
    new = chromosome + (np.random.rand(100)-0.5) * sigma
    return new

# We start by defining a population with candidates.
pop = Population(chromosomes=[random_start() for _ in range(5)],
                 eval_function=func_to_optimise, maximize=False)

# We define a sequence of steps to change these candidates
evo1 = (Evolution()
       .survive(fraction=0.2)
       .breed(parent_picker=pick_random_parents, combiner=make_child)
       # .mutate(mutate_function=add_noise, sigma=1))
       .mutate(mutate_function=add_noise, sigma=1))


# We define another sequence of steps to change these candidates
evo2 = (Evolution()
       .survive(n=1)
       .breed(parent_picker=pick_random_parents, combiner=make_child)
       # .mutate(mutate_function=add_noise, sigma=0.2))
       .mutate(mutate_function=add_noise, sigma=0.1))

# We are combining two evolutions into a third one. You don't have to
# but this approach demonstrates the flexibility of the library.
evo3 = (Evolution()
       .repeat(evo1, n=10)
       .evaluate())

# In this step we are telling evol to apply the evolutions
# to the population of candidates.
pop = pop.evolve(evo3, n=10000)
print(f"the best score found: {max([i.fitness for i in pop])}")

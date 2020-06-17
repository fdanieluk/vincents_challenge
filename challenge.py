import random
import requests as rq

import seaborn as sns
import matplotlib.pyplot as plt


# prepare request
url = "http://challenge.calmcode.io/attempt/"
emb = get_emb()

# make request, remember to use your email!
payload = {"user": "Filip Danieluk",
           "email": "filip.danieluk@cbre.com",
           "emb": emb}

print(payload)

resp = rq.post(url, json=payload)

# read response
print(resp.json())

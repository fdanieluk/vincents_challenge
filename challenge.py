import random
import requests as rq

import seaborn as sns
import matplotlib.pyplot as plt


# prepare request
url = "http://challenge.calmcode.io/attempt/"
emb = [random.random() for i in range(100)]

# make request, remember to use your email!
payload = {"user": "Filip Danieluk",
           "email": "filip.danieluk@cbre.com",
           "emb": emb}

print(payload)

sns.distplot(emb)
plt.show()
# resp = rq.post(url, json=payload)

# read response
# print(resp.json())

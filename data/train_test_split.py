import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

with open('data/jersey_number_labelling_via_project.json') as f:
    viaProject = json.load(f)

keys = list(viaProject["_via_img_metadata"].keys())
indices = np.arange(start=0, stop=len(keys), step=1)
placeholders = np.zeros(len(keys))
x_train, x_test, y_train, y_test = train_test_split(indices, placeholders, test_size=.2)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=.2)

football_train = list(map(lambda idx: viaProject["_via_img_metadata"][keys[idx]]["filename"], x_train))
football_val = list(map(lambda idx: viaProject["_via_img_metadata"][keys[idx]]["filename"], x_validation))
football_test = list(map(lambda idx: viaProject["_via_img_metadata"][keys[idx]]["filename"], x_test))

data = {}
data['train'] = football_train
data['num_train'] = len(football_train)
data['val'] = football_val
data['num_val'] = len(football_val)
data['test'] = football_test
data['num_test'] = len(football_test)

with open('data/football_train_test_split.json', 'w') as outfile:
    json.dump(data, outfile)
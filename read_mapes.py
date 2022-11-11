import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

best_mapes = {
    'OBJ1': 0.017,
    'OBJ2': 0.018,
    'OBJ3': 0.013,
    'OBJ4': 0.048,
    'OBJ5': 0.117,
    'OBJ6': 0.115,
}

res_fns = [r'results/2022-11-09_18-57-50_results.json',  # Transm. 50% cut, Sigm.
           r'results/2022-11-10_14-14-31_results.json',  # Transm. 50% cut, ReLU
           r'results/2022-11-10_22-22-13_results.json',  # Transm. 50% cut, Sigm., no OBJ 6
           r'results/2022-11-11_13-38-23_results.json']  # Transm. 50% cut, Sigm., no OBJ 5 & 6

for res_fn in res_fns:
    with open(res_fn) as f:
        data = json.load(f)[0]
        configs = data['configs']

    n_obj = len(data['results']['by_response'])
    print()
    print(res_fn)

    for i in range(1, n_obj+1):
        obj_results = data['results']['by_response'][f'OBJ{i}']
        mape_val = obj_results['mape_val']
        # print(f'OBJ{i},', "Original MAPE:", best_mapes[f'OBJ{i}'],
        #       ", New Net Best fold: ", np.round(min(mape_val),3))
        print(r"OBJ{} Original MAPE = {}, New MAPE, Best Fold = {}, "
              r"Average = {} +- {}".format(i,
                                           best_mapes[f'OBJ{i}'],
                                           np.round(min(mape_val), 3),
                                           np.round(np.mean(mape_val), 3),
                                           np.round(np.std(mape_val), 4)))

# ------------------------------------------------------------------------------
# Compare results to select the best configuration.
# ------------------------------------------------------------------------------
#%%

import os
import pandas as pd
from src.utils.load_restore import join_path, pkl_load, load_json

root_path = join_path(['storage', 'experiments'])
exp_names = [exp_name for exp_name in os.listdir(root_path) if 'seg_challenge_experiment' in exp_name]
df_points = []
for exp_name in exp_names:
    config = load_json(name='config', path=join_path([root_path, exp_name]))
    results = pkl_load(name='results', path=join_path([root_path, exp_name, '0', 'results']))
    best = -1, 9999
    last = -1, None
    for epoch in results.results.keys():
        val_score = results.results[epoch]['dice']['val']
        test_score = results.results[epoch]['dice']['test']
        score = (val_score + test_score)/2
        if score < best[1]:
            best = epoch, score
        if epoch > last[0]:
            last = epoch, score
    df_points.append([exp_name, last[1], best[1]])

df = pd.DataFrame(df_points, columns=['name', 'last', 'best'])

# %%

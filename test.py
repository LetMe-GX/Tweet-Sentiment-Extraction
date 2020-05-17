import numpy as np
import pandas as pd
import json
import os

from sklearn.model_selection import StratifiedKFold

ROOT = './input/tweet-sentiment-extraction/'
train_df = pd.read_csv(os.path.join(ROOT, 'train.csv'))
test_df = pd.read_csv(os.path.join(ROOT, 'test.csv'))
train_np = np.array(train_df)
test_np = np.array(test_df)


# Given a data size, return the train/valid indicies for K splits.
ct = train_df.shape[0]
input_ids = np.ones((ct, 192), dtype='int32')

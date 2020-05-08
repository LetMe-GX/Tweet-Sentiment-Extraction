# Copy predictions to submission file.
import json
import os

import pandas as pd

ROOT = './input/tweet-sentiment-extraction/'
test_df = pd.read_csv(os.path.join(ROOT, 'test.csv'))

def get_submission(post_processing):
    predictions = json.load(open('results/test_predictions.json', 'r'))
    submission = pd.read_csv(open('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv', 'r'))
    for i in range(len(submission)):
        id_ = submission['textID'][i]
        if post_processing and (test_df['sentiment'][i] == 'neutral' or len(test_df['text'][i].split()) <= 0): # post-processing
            submission.loc[i, 'selected_text'] = test_df['text'][i]
        else:
            submission.loc[i, 'selected_text'] = predictions[id_]

get_submission(False)
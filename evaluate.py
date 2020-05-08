import json

import numpy as np
from .preprocess import get_splits

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def evaluate(splits, np_train, post_processing=False):
    K = len(splits)
    predictions = [json.load(open('results/predictions_' + str(i+1) + '.json', 'r')) for i in range(K)]

    train_score = [{'neutral':[], 'positive':[], 'negative':[], 'total':[]} for _ in range(K+1)]
    valid_score = [{'neutral':[], 'positive':[], 'negative':[], 'total':[]} for _ in range(K+1)]

    for train_idx, line in enumerate(np_train):
        text_id = line[0]
        text = line[1]
        answer = line[2]
        sentiment = line[-1]

        if type(text) != str:
            continue

        for i, prediction in enumerate(predictions):
            if text_id not in prediction:
                print('key error:', text_id)
                continue
            else:
                if post_processing and (sentiment == 'neutral' or len(text.split()) <= 0): # post-processing
                    score = jaccard(answer, text)
                else:
                    score = jaccard(answer, prediction[text_id])

                if train_idx in splits[i]['valid_idx']:
                    valid_score[i][sentiment].append(score)
                    valid_score[i]['total'].append(score)
                    valid_score[K][sentiment].append(score)
                    valid_score[K]['total'].append(score)

                else:
                    train_score[i][sentiment].append(score)
                    train_score[i]['total'].append(score)
                    train_score[K][sentiment].append(score)
                    train_score[K]['total'].append(score)

    for i, score_dict in enumerate([train_score, valid_score]):
        if i == 0:
            print('train score \n')
        else:
            print('valid score \n')
        for j in range(K+1):
            for sentiment in ['neutral', 'positive', 'negative', 'total']:
                score = np.array(score_dict[j][sentiment])
                if j < K:
                    print('split', j+1)
                else:
                    print('all data')
                print(sentiment + ' - ' + str(len(score)) + ' examples, average score: ' + str(score.mean()))
            print()
import numpy as np
import pandas as pd
import json
import os

ROOT = './input/tweet-sentiment-extraction/'
train_df = pd.read_csv(os.path.join(ROOT, 'train.csv'))
test_df = pd.read_csv(os.path.join(ROOT, 'test.csv'))
train_np = np.array(train_df)
test_np = np.array(test_df)

# Given a data size, return the train/valid indicies for K splits.
K = 4
def split_data(num_examples, K=5):
    np.random.seed(0)
    idx = np.arange(num_examples)
    np.random.shuffle(idx)

    boundary = num_examples // K
    splits = [{} for _ in range(K)]
    for i in range(K):
        splits[i]['valid_idx'] = idx[i * boundary:(i + 1) * boundary]
        splits[i]['train_idx'] = np.concatenate((idx[:i * boundary], idx[(i + 1) * boundary:]))

        valid = train_np[splits[i]['valid_idx']]
        d = {'neutral': 0, 'positive': 0, 'negative': 0}
        for line in valid:
            d[line[-1]] += 1
        print(d)

    return splits


def get_splits():
    return split_data(len(train_np), K)


# Convert data to SQuAD-style
def convert_data(data, directory, filename):
    def find_all(input_str, search_str):
        l1 = []
        length = len(input_str)
        index = 0
        while index < length:
            i = input_str.find(search_str, index)
            if i == -1:
                return l1
            l1.append(i)
            index = i + 1
        return l1

    output = {}
    output['version'] = 'v1.0'
    output['data'] = []
    directory = os.path.join(ROOT, directory)

    for line in data:
        paragraphs = []
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(context) != str:
            print(context, type(context))
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context, 'qas': qas})
        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, filename), 'w') as outfile:
        json.dump(output, outfile)


def main():
    splits = split_data(len(train_np), K)

    # convert k-fold train data
    for i, split in enumerate(splits):
        data = train_np[split['train_idx']]
        directory = 'split_' + str(i + 1)
        filename = 'train.json'
        convert_data(data, directory, filename)

    # convert original train/test data
    data = train_np
    directory = 'original'
    filename = 'train.json'
    convert_data(data, directory, filename)

    data = test_np
    filename = 'test.json'
    convert_data(data, directory, filename)


if __name__ == "__main__":
    main()

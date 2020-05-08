import os

from qa.run_squad import run_squad
K = 5
ROOT = './input/tweet-sentiment-extraction/'

def run(cross_validation):
    if cross_validation:
        for i in range(1, K + 1):
            train_file = os.path.join(ROOT, "split_" + str(i) + "/train.json")
            predict_file = os.path.join(ROOT, "original/train.json")
            run_squad(train_file, predict_file, split=i)
    else:
        train_file = os.path.join(ROOT, "original/train.json")
        predict_file = os.path.join(ROOT, "original/test.json")
        run_squad(train_file, predict_file)


if __name__ == '__main__':
    run(cross_validation=False)
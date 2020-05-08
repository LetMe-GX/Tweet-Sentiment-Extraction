import os

from .input.qa.run_squad import main
K = 5
ROOT = './input/tweet-sentiment-extraction/'

def run(cross_validation):
    if cross_validation:
        for i in range(1, K + 1):
            train_file = os.path.join(ROOT, "split_" + str(i) + "/train.json")
            predict_file = os.path.join(ROOT, "original/train.json")
            main()
    else:
        train_file = os.path.join(ROOT, "original/train.json")
        predict_file = os.path.join(ROOT, "original/train.json")
        main()

if __name__ == 'main':
    run(cross_validation=False)
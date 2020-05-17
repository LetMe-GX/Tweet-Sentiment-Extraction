import collections

from qa.squad_metrics import normalize_answer


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def jaccard_evaluate(examples, preds):
    """
        Computes the jaccard scores from the examples and the model predictions
        """
    jaccard_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        if example.question_text == "neutral":
            prediction = example.context_text
        else:
            prediction = preds[qas_id]
        jaccard_scores[qas_id] = jaccard(gold_answers[0], prediction)

    total = len(jaccard_scores)
    scores = collections.OrderedDict(
        [
            ("jaccard", 100.0 * sum(jaccard_scores.values()) / total)
        ]
    )
    print("eval jaccard score:", scores)
    return scores

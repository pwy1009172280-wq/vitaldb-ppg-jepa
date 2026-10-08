"""Named sample/window and subject-level aggregation strategies."""

from collections import Counter, defaultdict

import numpy as np

from .probe import PredictionBatch


def aggregate_predictions(prediction: PredictionBatch, strategy: str) -> PredictionBatch:
    if strategy == "sample" or strategy == "window":
        return prediction
    if not prediction.subject_ids or len(prediction.subject_ids) != len(prediction.targets):
        raise ValueError("subject-level aggregation requires aligned subject_ids")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(prediction.subject_ids):
        groups[subject].append(index)
    subjects = tuple(groups)
    targets = []
    for indices in groups.values():
        values = prediction.targets[indices]
        if prediction.task == "classification" and len(set(values.tolist())) != 1:
            raise ValueError("classification targets must be consistent within subject")
        targets.append(values[0] if prediction.task == "classification" else float(np.mean(values)))
    if prediction.task == "classification":
        if prediction.probabilities is None:
            raise ValueError("classification aggregation requires probabilities")
        probabilities = []
        predicted = []
        scores = []
        for indices in groups.values():
            if strategy == "mean_probability":
                probability = prediction.probabilities[indices].mean(axis=0)
                score = prediction.scores[indices].mean(axis=0) if prediction.scores is not None else None
                predicted.append(int(probability.argmax()))
            elif strategy == "mean_score":
                if prediction.scores is None:
                    raise ValueError("mean_score requires classification scores")
                score = prediction.scores[indices].mean(axis=0)
                normalized = score - score.max()
                probability = np.exp(normalized)
                probability /= probability.sum()
                predicted.append(int(probability.argmax()))
            elif strategy == "majority_vote":
                if prediction.predicted_labels is None:
                    raise ValueError("majority_vote requires predicted_labels")
                votes = Counter(prediction.predicted_labels[indices].tolist())
                predicted.append(int(sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]))
                probability = np.bincount(prediction.predicted_labels[indices], minlength=len(prediction.class_vocabulary)) / len(indices)
                score = None
            else:
                raise ValueError("unknown classification aggregation strategy")
            probabilities.append(probability)
            scores.append(score)
        return PredictionBatch(np.asarray(targets), np.asarray(predicted), probabilities=np.asarray(probabilities),
                               scores=None if all(score is None for score in scores) else np.asarray(scores),
                               class_vocabulary=prediction.class_vocabulary, task=prediction.task, subject_ids=subjects)
    if strategy != "mean_prediction":
        raise ValueError("unknown regression aggregation strategy")
    scores = np.asarray([prediction.scores[indices].mean() for indices in groups.values()])
    return PredictionBatch(np.asarray(targets), scores=scores, task=prediction.task, subject_ids=subjects)

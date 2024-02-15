# See: https://github.com/surgical-vision/SAR_RARP50-evaluation/blob/main/sarrarp50/metrics/action_recognition.py  # noqa
import warnings

import numpy as np
from sklearn.metrics import accuracy_score


def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Yi_split


def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    return intervals


def segmental_confusion_matrix(P, Y, n_classes=0, bg_classes=None, overlap=0.1, **kwargs):
    def overlap_(p, y, n_classes, bg_classes, overlap):
        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        # Remove background labels
        if bg_classes is not None:
            if type(bg_classes) is int:
                bg_classes = [bg_classes]
            for bg_class in bg_classes:
                true_intervals = true_intervals[true_labels != bg_class]
                true_labels = true_labels[true_labels != bg_class]
                pred_intervals = pred_intervals[pred_labels != bg_class]
                pred_labels = pred_labels[pred_labels != bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TP, and FP.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j, 1], true_intervals[:, 1]) - np.maximum(
                pred_intervals[j, 0], true_intervals[:, 0]
            )
            union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) - np.minimum(
                pred_intervals[j, 0], true_intervals[:, 0]
            )
            IoU = (intersection / union) * (pred_labels[j] == true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1

        if bg_classes is not None:
            TP = np.delete(TP, bg_classes)
            FP = np.delete(FP, bg_classes)
        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()

        return TP, FP, FN

    if type(P) is list:
        return np.mean(
            [overlap_(P[i], Y[i], n_classes, bg_classes, overlap) for i in range(len(P))]
        )
    else:
        return overlap_(P, Y, n_classes, bg_classes, overlap)


def segmental_f1score(P, Y, n_classes=0, bg_classes=None, overlap=0.1, **kwargs):
    TP, FP, FN = segmental_confusion_matrix(P, Y, n_classes, bg_classes, overlap)
    return 2 * TP / (2 * TP + FP + FN)


def segmental_precision_recall(P, Y, n_classes=0, bg_classes=None, overlap=0.1, **kwargs):
    TP, FP, FN = segmental_confusion_matrix(P, Y, n_classes, bg_classes, overlap)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


def frame_accuracy(P, Y, **kwargs):
    def acc_(p, y):
        return np.mean(p == y)

    if type(P) is list:
        return np.mean([np.mean(P[i] == Y[i]) for i in range(len(P))])
    else:
        return acc_(P, Y)


def frame_confusion_matrix(P, Y, n_classes, ignore_classes=None):
    # 混同行列の計算
    cm = np.zeros((n_classes, n_classes))
    for i in range(len(Y)):
        cm[Y[i], int(P[i])] += 1
    if ignore_classes is not None:
        cm = np.delete(cm, ignore_classes, axis=0)
        cm = np.delete(cm, ignore_classes, axis=1)
    # TP, FP, FNの計算
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)  # is it correct?
    return TP, FP, FN, TN


def calc_precision(TP, FP, FN, class_wise=False):
    precision = _safe_divide(TP, TP + FP)
    return precision if class_wise else np.mean(precision)


def calc_recall(TP, FP, FN, class_wise=False):
    recall = _safe_divide(TP, TP + FN)
    return recall if class_wise else np.mean(recall)


def calc_f1score(TP, FP, FN, class_wise=False):
    f1score = _safe_divide(2 * TP, 2 * TP + FP + FN)
    return f1score if class_wise else np.mean(f1score)


def _safe_divide(x, y):
    return np.divide(x, y, out=np.zeros_like(x), where=y != 0).tolist()


def frame_score_report(P, Y, n_classes, class_names=None, ignore_classes=None):
    if len(P) != len(Y):
        warnings.warn(f"Prediction and ground truth have different lengths: {len(P)} vs {len(Y)}")
        num_samples = min(len(P), len(Y))
        P = P[:num_samples].astype(int)
        Y = Y[:num_samples].astype(int)

    tp, fp, fn, tn = frame_confusion_matrix(P, Y, n_classes, ignore_classes=ignore_classes)
    precision = calc_precision(tp, fp, fn, class_wise=True)
    recall = calc_recall(tp, fp, fn, class_wise=True)
    f1score = calc_f1score(tp, fp, fn, class_wise=True)
    if ignore_classes is not None:
        P_filtered = []
        Y_filtered = []
        for p, y in zip(P, Y):
            if y not in set(ignore_classes):
                P_filtered.append(p)
                Y_filtered.append(y)
        P = np.array(P_filtered)
        Y = np.array(Y_filtered)
    accuracy = accuracy_score(P, Y)
    report = {
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
        "accuracy": accuracy,
        "mean_precision": np.mean(precision),
        "mean_recall": np.mean(recall),
        "mean_f1score": np.mean(f1score),
    }
    if class_names is not None:
        report["original_class_names"] = class_names
        if ignore_classes is not None:
            class_names = [
                class_name for i, class_name in enumerate(class_names) if i not in ignore_classes
            ]
        report["class_names"] = class_names
    return report


def segment_score_report(P, Y, n_classes, class_names=None, ignore_classes=None, overlap=0.1):
    if len(P) != len(Y):
        warnings.warn(f"Prediction and ground truth have different lengths: {len(P)} vs {len(Y)}")
        num_samples = min(len(P), len(Y))
        P = P[:num_samples].astype(int)
        Y = Y[:num_samples].astype(int)

    tp, fp, fn = segmental_confusion_matrix(
        P, Y, n_classes, bg_classes=ignore_classes, overlap=overlap
    )
    precision = calc_precision(tp, fp, fn, class_wise=True)
    recall = calc_recall(tp, fp, fn, class_wise=True)
    f1score = calc_f1score(tp, fp, fn, class_wise=True)
    report = {
        f"segmental_precision@{int(overlap*100)}": precision,
        f"segmental_recall@{int(overlap*100)}": recall,
        f"segmental_f1score@{int(overlap*100)}": f1score,
    }
    if class_names is not None:
        report["original_class_names"] = class_names
        if ignore_classes is not None:
            class_names = [
                class_name for i, class_name in enumerate(class_names) if i not in ignore_classes
            ]
        report["class_names"] = class_names
    return report

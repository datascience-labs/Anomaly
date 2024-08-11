import numpy as np
import more_itertools as mit
import pickle
import pandas as pd
import torch
from spot import SPOT, dSPOT

from args import get_parser
parser = get_parser()
args = parser.parse_args()

def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    global ano_sc, lla, thre
    ano_sc = score
    lla = label
    thre = threshold

    if (label is None) | (args.adjust == True):
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if actual[i].any() and predict[i].any() and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j].any():
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_point2point(predict, actual):

    global preds, acts

    preds = predict
    acts = actual
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def pot_eval(init_score, score, label, q=1e-3, level=0.99, dynamic=False):

    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # Calibration step
    ret = s.run(dynamic=dynamic, with_alarm=False)

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": pot_th,
            "latency": p_latency,
        }
    else:
        return {
            "threshold": pot_th,
        }


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):

    print(f"Finding best f1-score by searching for threshold..")
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target, latency = calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "TP": int(m[3]),
        "TN": int(m[4]),
        "FP": int(m[5]),
        "FN": int(m[6]),
        "threshold": m_t,
        "latency": m_l,
    }


def calc_seq(score, label, threshold):
    predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    return calc_point2point(predict, label), latency


def epsilon_eval(train_scores, test_scores, test_labels, reg_level=1):
    print('train_scores: ', train_scores.shape)
    best_epsilon = find_epsilon(train_scores, reg_level)
    print('best_epsilon: ', best_epsilon.shape)

    pred, p_latency = adjust_predicts(test_scores, test_labels, best_epsilon, calc_latency=True)
    if test_labels is not None:
        p_t = calc_point2point(pred, test_labels)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": best_epsilon,
            "latency": p_latency,
            "reg_level": reg_level,
        }
    else:
        return {"threshold": best_epsilon, "reg_level": reg_level}


def find_epsilon(errors, reg_level=1):

    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = Fscore
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)

    print('best_epsilon in : ', best_epsilon.shape)
    return best_epsilon


def rca(path, group):
    with open(f'{path}/test_output.pkl', 'rb') as f:
        x = pickle.load(f)
    b = []
    for i in range(38):
        a = x[f"A_Score_{i}"]
        b.append(list(a))

    b = torch.tensor(b)
    root_causes, root_cause_index = torch.topk(b.permute(1, 0), 38)

    root_cause_label = pd.read_csv(f'/data/SMD/interpretation_label/machine-{group}.txt', header=None, delimiter='\t')

    root_cause_label[['section', 'root cause']] = root_cause_label[0].str.split(':', 1, expand=True)
    root_cause_label.drop(0, axis=1, inplace=True)

    root_cause_index = root_cause_index[100+1:-1]

    hitrate_100 = get_hitrate(x, root_cause_index, root_cause_label, 1)
    hitrate_150 = get_hitrate(x, root_cause_index, root_cause_label, 1.5)

    return {
        "group": group,
        "hitrate_100": hitrate_100,
        "hitrate_150": hitrate_150,
    }

def get_hitrate(x, root_cause_index, root_cause_label, ratio):
    hit = []
    gt_count = []
    for x in range(root_cause_label.shape[0]):
        temp = []
        for i in range(int(root_cause_label.iloc[x, 0].split('-')[0]), int(root_cause_label.iloc[x, 0].split('-')[1])+1):
            gt = list(map(int, root_cause_label.iloc[x, 1].split(',')))
            gt_num = len(gt)

            candidates = root_cause_index[i, :round(gt_num * ratio)]
            count_matching_values = len(np.intersect1d(np.array(gt), np.array(candidates)))

            temp.append(count_matching_values/ gt_num)
        gt_count.append(gt_num)
        hit.append(np.mean(temp))

    return np.mean(hit)

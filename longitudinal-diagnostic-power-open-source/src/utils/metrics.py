import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from math import sqrt




def return_cases_from_cm(tp=0, tn=0, fp=0, fn=0):
    return tp + fn


def return_non_cases_from_cm(tp=0, tn=0, fp=0, fn=0):
    return tn + fp


def return_prevalence_cases_from_cm(tp=0, tn=0, fp=0, fn=0):
    if (tp + tn + fp + fn) == 0:
        return 0
    return (tp + fn) / (tp + tn + fp + fn)


def return_prevalence_non_cases_from_cm(tp=0, tn=0, fp=0, fn=0):
    if (tp + tn + fp + fn) == 0:
        return 0
    return (tn + fp) / (tp + tn + fp + fn)


def return_tp_from_cm(tp=0, tn=0, fp=0, fn=0):
    return tp


def return_tn_from_cm(tp=0, tn=0, fp=0, fn=0):
    return tn


def return_fp_from_cm(tp=0, tn=0, fp=0, fn=0):
    return fp


def return_fn_from_cm(tp=0, tn=0, fp=0, fn=0):
    return fn


def calculate_accuracy_from_cm(tp=0, tn=0, fp=0, fn=0):
    total = tn + fp + fn + tp
    return (tp + tn) / total if total > 0 else 0.0


def calculate_recall_from_cm(tp=0, tn=0, fp=0, fn=0):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def calculate_precision_from_cm(tp=0, tn=0, fp=0, fn=0):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calculate_specificity_from_cm(tp=0, tn=0, fp=0, fn=0):
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def calculate_fpr_from_cm(tp=0, tn=0, fp=0, fn=0):
    return fp / (tn + fp) if (tn + fp) > 0 else 0.0


def calculate_fnr_from_cm(tp=0, tn=0, fp=0, fn=0):
    return fn / (tp + fn) if (tp + fn) > 0 else 0.0


def calculate_f1_beta_score_from_cm(tp=0, tn=0, fp=0, fn=0, beta=1):
    if (tp + fp + fn) == 0:
        return np.nan
    precision = calculate_precision_from_cm(tp=tp, fp=fp)
    recall = calculate_recall_from_cm(tp=tp, fn=fn)
    return (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall) if (precision + recall) > 0 else 0.0


def calculate_f1_score_from_cm(tp=0, tn=0, fp=0, fn=0):
    return calculate_f1_beta_score_from_cm(tp=tp, fp=fp, fn=fn)



def calculate_adjusted_f1_from_cm(tp=0, tn=0, fp=0, fn=0, target_prevalence=0.01):
    """Implementing Rueben's prevalence adjusted f1 score"""
    total = tp + tn + fp + fn
    if total == 0:
        # If there are no samples, return 0 as the F1 score is undefined
        return 0

    # Case prevalence of nt1
    case_prevalence = (tp + fn) / total

    # Prevent division by zero with a small epsilon
    epsilon = 1e-7
    tp_fn_adjustment = target_prevalence / (case_prevalence + epsilon)
    tn_fp_adjustment = (1 - target_prevalence) / (1 - case_prevalence + epsilon)

    # Apply adjustments
    tp_adjusted = tp * tp_fn_adjustment
    fn_adjusted = fn * tp_fn_adjustment
    fp_adjusted = fp * tn_fp_adjustment

    # Calculate adjusted F1 score
    precision_adjusted = tp_adjusted / (tp_adjusted + fp_adjusted) if (tp_adjusted + fp_adjusted) > 0 else 0
    recall_adjusted = tp_adjusted / (tp_adjusted + fn_adjusted) if (tp_adjusted + fn_adjusted) > 0 else 0
    f1_adjusted = 2 * (precision_adjusted * recall_adjusted) / (precision_adjusted + recall_adjusted) if (precision_adjusted + recall_adjusted) > 0 else 0

    return f1_adjusted


def calculate_balanced_accuracy_from_cm(tp=0, tn=0, fp=0, fn=0):
    """
    Calculates the Balanced Accuracy from a confusion matrix.
    Balanced Accuracy = (Sensitivity + Specificity) / 2
    """
    recall = calculate_recall_from_cm(tp=tp, fn=fn) 
    specificity = calculate_specificity_from_cm(tn=tn, fp=fp)
    balanced_accuracy = (recall + specificity) / 2
    return balanced_accuracy


def calculate_mcc_from_cm(tp=0, tn=0, fp=0, fn=0):
    """
    Calculates the Matthews Correlation Coefficient (MCC) from a confusion matrix.
    MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    """
    numerator = tp * tn - fp * fn
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator > 0 else 0


def calculate_recall_given_minimum_specificity_from_cm(tp=0, tn=0, fp=0, fn=0, minimum_specificity=0.995):

    # Ensure specificity is above minimum
    specificity = calculate_specificity_from_cm(tp=tp, tn=tn, fp=fp, fn=fn)
    if specificity < minimum_specificity:
        return 0.0

    # return recall if condition is fulfilled:
    return calculate_recall_from_cm(tp=tp, tn=tn, fp=fp, fn=fn)


def calculate_f1_given_minimum_specificity_from_cm(tp=0, tn=0, fp=0, fn=0, minimum_specificity=0.995, beta=1):

    # Ensure specificity is above minimum
    specificity = calculate_specificity_from_cm(tp=tp, tn=tn, fp=fp, fn=fn)
    if specificity < minimum_specificity:
        return 0.0

    # return recall if condition is fulfilled:
    
    return calculate_f1_beta_score_from_cm(tp=tp, tn=tn, fp=fp, fn=fn, beta=beta)


def calculate_specificity_given_minimum_recall_from_cm(tp=0, tn=0, fp=0, fn=0, minimum_recall=0.95):

    # Ensure specificity is above minimum
    recall = calculate_recall_from_cm(tp=tp, tn=tn, fp=fp, fn=fn)
    if recall < minimum_recall:
        return 0.0

    # return recall if condition is fulfilled:
    return calculate_specificity_from_cm(tp=tp, tn=tn, fp=fp, fn=fn)


def calculate_metrics_from_array(y_true, y_pred, pos_label=1):
    assert y_true.shape == y_pred.shape
    out = {
        'tp': sum((y_true == pos_label) & (y_pred == pos_label)),
        'fp': sum((y_true != pos_label) & (y_pred == pos_label)),
        'fn': sum((y_true == pos_label) & (y_pred != pos_label)),
        'tn': sum((y_true != pos_label) & (y_pred != pos_label))
    }
    return out


def calculate_accuracy_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_accuracy_from_cm(**out)


def calculate_true_positives_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return return_tp_from_cm(**out)


def calculate_false_positives_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return return_tp_from_cm(**out)


def calculate_false_negatives_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return return_tp_from_cm(**out)


def calculate_true_negatives_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return return_tp_from_cm(**out)


def calculate_recall_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_recall_from_cm(**out)


def calculate_precision_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_precision_from_cm(**out)


def calculate_specificity_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_specificity_from_cm(**out)


def calculate_fpr_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_fpr_from_cm(**out)


def calculate_fnr_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_fnr_from_cm(**out)


def calculate_fbeta_score_from_array(y_true, y_pred, pos_label=1, beta=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_f1_beta_score_from_cm(beta=beta, **out)


def calculate_f1_score_from_array(y_true, y_pred, pos_label=1):
    return calculate_fbeta_score_from_array(y_true=y_true, y_pred=y_pred, pos_label=pos_label)


def calculate_kappa_multi_from_array(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    cohens_kappa = cohen_kappa_score(y_true, y_pred)
    return cohens_kappa


def calulate_roc_auc_score(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    assert y_true.shape == y_pred.shape
    roc_auc = roc_auc_score(y_true, y_pred)
    return roc_auc



def calculate_adjusted_f1_from_array(y_true, y_pred, target_prevalence=0.01, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_adjusted_f1_from_cm(target_prevalence=target_prevalence, **out)


def calculate_balanced_accuracy_from_array(y_true, y_pred, pos_label=1):
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_balanced_accuracy_from_cm(**out)


def calculate_mcc_from_array(y_true, y_pred, pos_label=1): 
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_mcc_from_cm(**out)


def calculate_recall_given_minimum_specificity_from_array(y_true, y_pred, minimum_specificity=0.995, pos_label=1): 
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_recall_given_minimum_specificity_from_cm(minimum_specificity=minimum_specificity, **out)


def calculate_recall_given_minimum_specificity_wrapper(minimum_specificity=0.995, pos_label=1):
    def calculate_recall_given_minimum_specificity_from_array_inner(y_true, y_pred): 
        out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
        return calculate_recall_given_minimum_specificity_from_cm(minimum_specificity=minimum_specificity, **out)
    return calculate_recall_given_minimum_specificity_from_array_inner


def calculate_f1_given_minimum_specificity_wrapper(minimum_specificity=0.995, pos_label=1, beta=1):
    def calculate_f1_given_minimum_specificity_from_array_inner(y_true, y_pred): 
        out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
        return calculate_f1_given_minimum_specificity_from_cm(minimum_specificity=minimum_specificity, beta=beta, **out)
    return calculate_f1_given_minimum_specificity_from_array_inner


def calculate_specificity_given_minimum_recall_from_array(y_true, y_pred, minimum_recall=0.95, pos_label=1): 
    out = calculate_metrics_from_array(y_true, y_pred, pos_label=pos_label)
    return calculate_specificity_given_minimum_recall_from_cm(minimum_recall=minimum_recall, **out)


def calculate_aucpr(y_true, y_pred):
    from sklearn.metrics import precision_recall_curve, auc    
    if len(np.unique(y_true)) < 2:
        return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


if __name__ == '__main__':
    
    # help(prg)

    N = 1000
    np.random.seed(42)
    tar = np.random.randint(0, 2, size=(N))
    pre = np.random.randint(0, 2, size=(N))

    data_prevalence = (tar == 1).mean()

    #print(f"{tar=}")
    #print(f"{pre=}")
    print(f"{data_prevalence=}")

    # compute performance
    #f1_adj = adjusted_f1(y_pred=pre, y_true=tar)
    #f1 = calculate_f1_score(y_true=tar, y_pred=pre, pos_label=1)
    #print(f"f1_adj is: {f1_adj}")
    #print(f"f1 is: {f1}")

    # new gain metrics
    out = calculate_metrics_from_array(y_true=tar, y_pred=pre, pos_label=1)
    print(out)

    #preG = calculate_precision_gain_from_cm_yours(**out)
    #recG = calculate_recall_gain_from_cm_yours(**out)
    prec = calculate_precision_from_cm(**out)
    recall = calculate_recall_from_cm(**out)
    #recGO = calculate_recall_gain_from_cm(**out)
    #preGO = calculate_precision_gain_from_cm(**out)

    # other 
    print(tar.shape)
    print(pre.shape)

    aucroc = calulate_roc_auc_score(tar, pre)
    aucpr = calculate_aucpr(tar, pre)
    #aucprg = calculate_aucprg(tar, pre)
    
    print('pre', prec)
    print('re', recall)
    #print('preG', preG)
    #print('reG', recG)
    #print('preGO', preGO)
    #print('reGO', recGO)
    
    print(f"{aucroc=}")
    print(f"{aucpr=}")
    #print(f"{aucprg=}")
    
    # print(f"accuracy is: {metrics['accuracy'](tar, pre)}")

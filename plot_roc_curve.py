# *--encoding:utf-8--*
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os

def plot_single_roc_curve(ys, preds):
    # print len(item_ids_temp)
    fpr, tpr, threshold = roc_curve(ys, preds)
    auc = roc_auc_score(ys, preds)

    plt.plot(fpr, tpr, color='g', linewidth=3, label='AUC = %.3f' % auc)
    plt.legend(loc='lower right')
    # plt.set_title('\"%s\" %s by %s roc curve' % (CLASS_NAME_DICT[c], p_or_i, p_or_i))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axvline(x=0.08, color='r', linestyle='--')
    plt.axhline(y=0.995, color='r', linestyle='--')
    plt.show()
    pass


def calculate_AUC(label_dict, prob_dict, classes):
    AUC_dict = dict()
    for c in classes:
        item_ids = prob_dict.keys()
        ys = [label_dict[iid][c] for iid in item_ids]
        preds = [prob_dict[iid][c] for iid in item_ids]
        if len(set(ys)) == 1:
            print 'warning: only one class present in y_true , ROC AUC score, set auc as -1'
            AUC_dict[c] = -1
        else:
            AUC_dict[c] = roc_auc_score(ys, preds)

    return AUC_dict


def plot_multiple_roc_curves(label_dict, prob_dict, classes_names, output_dir):
    fig, axes = plt.subplots(len(classes_names) - 1, 1, figsize=(8, 32))
    item_ids = prob_dict.keys()
    AUC_dict = calculate_AUC(label_dict, prob_dict, range(len(classes_names)))

    for c, c_name in enumerate(classes_names):
        if c == 0:
            continue  # pass for normal

        ys = [label_dict[iid][c] for iid in item_ids]
        preds = [prob_dict[iid][c] for iid in item_ids]
        fprs, tprs, thresh = roc_curve(ys, preds)

        axes[c - 1].set_title("Item-wise ROC Curve: {}".format(c_name))
        axes[c - 1].plot(fprs, tprs, 'g', label="AUC=%0.2f" % AUC_dict[c])
        axes[c - 1].legend(loc='lower right')
        axes[c - 1].plot([0, 1], [0.995, 0.995], 'r--')
        axes[c - 1].plot([0.08, 0.08], [0, 1], 'r--')
        axes[c - 1].set_xlim([0.0, 1.0])
        axes[c - 1].set_ylim([0.0, 1.0])
        axes[c - 1].set_ylabel("TPR ( = 1 - FNR)")
        axes[c - 1].set_xlabel("FPR")

    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))



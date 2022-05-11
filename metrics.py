import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

class PrintColors:

    GREEN = "\033[0;32m"
    BLUE = "\033[1;34m"
    RED = "\033[1;31m"

    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END_COLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MEL = 0  # Melanoma
NV = 1  # Melanocytic nevus
BCC = 2  # Basal cell carcinoma
AKIEC = 3  # Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
BKL = 4  # Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
DF = 5  # Dermatofibroma
VASC = 6  # Vascular lesion

classes = [MEL, NV, BCC, AKIEC, BKL, DF, VASC]
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

def get_confusion_matrix(y_true, y_pred, norm_cm=True, print_cm=True):
    if y_true.dim() > 1:
        true_class = np.argmax(y_true, axis=1)
        pred_class = np.argmax(y_pred, axis=1)
    else:
        true_class = y_true.cpu().numpy()
        pred_class = y_pred.cpu().numpy()

    cnf_mat = confusion_matrix(true_class, pred_class, labels=classes)

    total_cnf_mat = np.zeros(shape=(cnf_mat.shape[0] + 1, cnf_mat.shape[1] + 1), dtype=np.float)
    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    for i_row in range(cnf_mat.shape[0]):
        total_cnf_mat[i_row, -1] = np.sum(total_cnf_mat[i_row, 0:-1])

    for i_col in range(cnf_mat.shape[1]):
        total_cnf_mat[-1, i_col] = np.sum(total_cnf_mat[0:-1, i_col])

    if norm_cm:
        cnf_mat = cnf_mat/(cnf_mat.astype(np.float).sum(axis=1)[:, np.newaxis] + 0.001)

    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    if print_cm:
        print_confusion_matrix(cm=total_cnf_mat, labels=class_names + ['TOTAL', ])

    return cnf_mat


def get_precision_recall(y_true, y_pred, print_pr=True):
    if y_true.dim() > 1:
        true_class = np.argmax(y_true, axis=1)
        pred_class = np.argmax(y_pred, axis=1)
    else:
        true_class = y_true.cpu().numpy()
        pred_class = y_pred.cpu().numpy()

    precision, recall, _, _ = precision_recall_fscore_support(y_true=true_class,
                                                              y_pred=pred_class,
                                                              labels=classes,
                                                              warn_for=())
    if print_pr:
        print_precision_recall(precision=precision, recall=recall, labels=class_names)

    return precision, recall

def print_confusion_matrix(cm, labels):
    """pretty print for confusion matrixes"""

    columnwidth = max([len(x) for x in labels] + [12])
    # Print header
    print()
    first_cell = r"True\Pred"
    print("|%{0}s|".format(columnwidth - 2) % first_cell, end="")
    for label in labels:
        print("%{0}s|".format(columnwidth -1) % label, end="")
    print()

    first_cell = "-------"
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for _ in labels:
        print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print()

    # Print rows
    for i, label1 in enumerate(labels):
        print("|%{0}s|".format(columnwidth - 2) % label1, end="")
        for j in range(len(labels)):
            cell = "%{0}.2f|".format(columnwidth-1) % cm[i, j]
            if i == len(labels) - 1 or j == len(labels) - 1:
                cell = "%{0}d|".format(columnwidth-1) % cm[i, j]
                if i == j:
                    print("%{0}s|".format(columnwidth-1) % ' ', end="")
                else:
                    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")
            elif i == j:
                print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")
            else:
                print(PrintColors.RED + cell + PrintColors.END_COLOR, end="")

        print()


def print_precision_recall(precision, recall, labels):
    columnwidth = max([len(x) for x in labels] + [12])
    # Print header
    print()
    first_cell = " "
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for label in labels:
        print("%{0}s|".format(columnwidth-1) % label, end="")
    print("%{0}s|".format(columnwidth-1) % 'MEAN', end="")
    print()

    first_cell = "-------"
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for _ in labels:
        print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print()

    # print precision
    print("|%{0}s|".format(columnwidth-2) % 'precision', end="")
    for j in range(len(labels)):
        cell = "%{0}.3f|".format(columnwidth-1) % precision[j]
        print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")

    cell = "%{0}.3f|".format(columnwidth-1) % np.mean(precision)
    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")

    print()

    # print recall
    print("|%{0}s|".format(columnwidth-2) % 'recall', end="")
    for j in range(len(labels)):
        cell = "%{0}.3f|".format(columnwidth-1) % recall[j]
        print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")

    cell = "%{0}.3f|".format(columnwidth-1) % np.mean(recall)
    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")

    print('')

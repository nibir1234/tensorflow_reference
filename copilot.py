# ====================================================== Multiclass ======================================================

# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score

def calculate_multiclass_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results


# ================================================ Multiclass Separate ===================================================
#  ref ==> https://colab.research.google.com/drive/1i8aGlH3LsvhgmnxvBwu7BbGU4a2_46fi#scrollTo=of34o7VUamGc
#  Precision, Recall, F1 spearate for each class
#         precision ==>  {'neutral': 0.6746987951807228, 'positive': 0.5, 'negative': 0.4594594594594595}
#         recall    ==>  {'neutral': 0.8484848484848485, 'positive': 0.18269230769230768, 'negative': 0.4}
#         f1        ==>  {'neutral': 0.7516778523489932, 'positive': 0.2676056338028169, 'negative': 0.42767295597484273}
# ========================================================================================================================


def calculate_multiclass_results_separate_prf1(y_true, y_pred, score_func):

  model_score_all = score_func(y_true, y_pred, average=None)
  model_score_class = {
                  "neutral": model_score_all[0],
                  "positive": model_score_all[1],
                  "negative": model_score_all[2]
  }
  return model_score_class

# model_6_results_r = calculate_multiclass_results_separate_prf1( val_labels_one_hot, model_6_preds, recall_score)
# print("recall    ==> ", model_6_results_r)

# ========================================================================================================================


def compare_models(model_1_results, baseline_results):
  import numpy as np
  return np.array(list(model_1_results.values())) > np.array(list(baseline_results.values()))


# ========================================================================================================================




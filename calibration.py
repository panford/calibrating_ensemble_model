import numpy as np
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

class calibrator():
  def __init__(self, class_type='binary', n_bins=10):
    self.class_type = class_type
    self.n_bins = None
    self.results_dict = {}

    if class_type == "binary":
      results_dict['ece'] = self.expected_cal_error_two_classes()
      results_dict['mce'] = self.maximimum_cal_error_two_classes()
      results_dict['rmsce'] = self.rmsce_cal_error_two_classes()

    elif class_type == 'multiclass':
      results_dict['ece'] = self.expected_cal_error_multiclass()
      results_dict['mce'] = self.maximimum_cal_error_multiclass()
      results_dict['rmsce'] = self.rmsce_cal_error_multiclass()

    self.dataframe = None

  def get_prob_scores(self, y_pred, y_true):
    return calibration_curve(y_val_binary, y_pred_binary, self.n_bins)

  def expected_cal_error_two_classes(self, true_probs, pred_probs, bin_ranges):

    ecetc = 0
    for bs in np.arange(len(bin_sizes)):
      ecetc += (bin_ranges[bs] / sum(bin_ranges)) * np.abs(prob_true[bs] - prob_pred[bs])
    return ecetc
    

  def maximimum_cal_error_two_classes(self, prob_true, prob_pred, bin_ranges):
      mcetc = 0
      for bs in np.arange(len(bin_ranges)):
          mcetc = max(mcetc, np.abs(prob_true[bs] - prob_pred[bs]))
      return mcetc

  def rmsce_cal_error_two_classes(self, prob_true, prob_pred, bin_ranges):
      rmscetc = 0
      for bs in np.arange(len(bin_ranges)):
          rmscetc += (bin_ranges[bs] / sum(bin_ranges)) * (prob_true[bs] - prob_pred[bs]) ** 2
      return np.sqrt(rmsce)


  def expected_cal_error_multiclass(self, y_true, y_pred):

      ece_bin = []
      for a_class in range(y_true.shape[1]):
          prob_true, prob_pred = calibration_curve(y_true[a_class], y_pred[a_class], n_bins=10)
          bin_sizes = np.histogram(a=y_pred[a_class], range=(0, 1), bins=len(prob_true))[0]
          ece_bin.append(self.ece_calculation_binary(prob_true, prob_pred, bin_sizes))

      return np.mean(ece_bin)
          
      
  def maximimum_cal_error_multiclass(self, y_true, y_pred):
      mce_bin = []
      for a_class in range(y_true.shape[1]):
          prob_true, prob_pred = calibration_curve(y_true[a_class], y_pred[a_class], n_bins=10)
          bin_sizes = np.histogram(a=y_pred[a_class], range=(0, 1), bins=len(prob_true))[0]
          mce_bin.append(self.mce_calculation_binary(prob_true, prob_pred, bin_sizes))

      return np.mean(mce_bin)
      
  def rmsce_cal_error_multiclass(self, y_true, y_pred):
      rmsce_bin = []
      for a_class in range(y_true.shape[1]):
          prob_true, prob_pred = calibration_curve(y_true[a_class], y_pred[a_class], n_bins=10)
          bin_sizes = np.histogram(a=y_pred[a_class], range=(0, 1), bins=len(prob_true))[0]
          rmsce_bin.append(self.rmsce_calculation_binary(prob_true, prob_pred, bin_sizes))

      return np.mean(rmsce_bin)


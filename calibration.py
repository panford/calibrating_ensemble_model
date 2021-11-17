import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt



class Calibration(nn.Module):
  def __init__(self, num_bins, prob_range, num_classes=2):
    self.num_bins = num_bins
    self.prob_range = prob_range

  def get_accuracy_confidence(self, probabilities, predictions, targets, eps = 10e-4):

  
    _, bin_edges = torch.histogram(probabilities, bins=self.num_bins, range=self.prob_range)
    
    bin_idx = torch.bucketize(probabilities, bin_edges)
    bin_counts = torch.bincount(bin_idx, minlength = self.num_bins).add(eps)

    confidence_vector = bin_idx.bincount(confidence, self.num_bins).div(bin_counts)
    accuracy_vector = bin_idx.bincount(targets, self.num_bins).div(bin_counts)

    return accuracy_vector, confidence_vector, bin_counts
  

  def Binary_ECE(self, probabilities, predictions, targets, norm=1): 
    """
    Expected Calibration Error for Binary class 
    """

    assert norm == 1 or norm == 2, "norm should be either 1 or 2"
    accuracies, confidences, bin_counts = self.get_accuracy_confidence(probabilities, predictions, targets)
    acc_conf_diff = accuracies - confidences
    coeffs = bin_counts/ len(probabilities)
    scaled_diff = coeffs * acc_conf_diff
    normed_diff = torch.linalg.norm(scaled_diff, norm)
    return np.sum(np.abs(normed_diff.numpy()))


  def Binary_MCE(self, probabilities, predictions, targets):
    """
    Maximum Calibration Error for Binary class
    """
    accuracies, confidences, _ = self.get_accuracy_confidence(probabilities, predictions, targets)
    acc_conf_diff = torch.abs(accuracies - confidences)
    return np.amax(acc_conf_diff.numpy(), axis=0).item()

  def Binary_RMSCE(self, probabilities, predictions, targets):
    """Binary Root Mean Squared Calibration Error"""
    accuracies, confidences, bin_counts = self.get_accuracy_confidence(probabilities, predictions, targets)
    acc_conf_diff = np.square(accuracies - confidences)
    coeffs = bin_counts/ len(probabilities)
    scaled_diff = coeffs * acc_conf_diff
    return np.sqrt(np.sum(acc_conf_diff.numpy())).item()

  #TODO Multiclass Calibration Errors
  # def Multiclass_ECE(self, probabilities, predictions, targets):
  #   return 

  def reliability_diagram(self, probabilities, predictions, targets, model_name, ax=None):
    
    accuracies, confidences, _ = self.get_accuracy_confidence(probabilities, predictions, targets)
    if ax==None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
    else:
        plt.sca(ax)
    
    plt.plot([0, 1], [0, 1], color="#FE4A49", linestyle=":", label="Perfect calibration")
    plt.plot(accuracies, confidences, "s-", label=model_name, color="#162B37")

    plt.ylabel("Accuracies", fontsize=12)
    plt.xlabel("Confidences", fontsize=12,)

    plt.legend(fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.grid(True, color="#B2C7D9")

    plt.tight_layout()

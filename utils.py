import torch
import os

def save_checkpoint(path, kwargs):
  if not os.path.exists(path):
    os.makedirs(path)
  torch.save(kwargs, os.path.join(path, 'ensemble.pt'))
  # else:

def load_checkpoint(path):
  return torch.load(os.path.join(path, 'ensemble.pt'))

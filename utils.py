import os
import torch
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(path, kwargs):
  file_ = os.path.join(path, 'ensemble.pt')

  if os.path.isfile(file_):
    os.remove(file_)

  if not os.path.exists(path):
    os.makedirs(path)
    
  torch.save(kwargs, file_)
  # else

def load_checkpoint(path):
  file_ = os.path.join(path, 'ensemble.pt')
  if not os.path.exists(file_):
    raise FileNotFoundError ("Checkpoint file not present")
  else: 
    return torch.load(file_)

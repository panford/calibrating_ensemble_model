import torch.nn as nn
import torch.nn.functional as F
class ResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.resnet = resnet18(pretrained=False, num_classes=num_classes)

        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(2 * 2 * 128, 256)

    def compute_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.flatten(1)

        x = F.relu(self.fc1(x))

        return x

class EnsembleModel(nn.Module):
  def __init__(self, emsemble_size, model):
    super(EnsembleModel, self).__init__()
    self.ensemble_size = ensemble_size

    self.ensemble = nn.ModuleList(model for model in range(ensemble_size))

  def forward(self, x):
    x = self.ensemble(x)


class SoftmaxModel(Model):
  def __init__(self, input_size, num_classes):
      super().__init__()

      self.last_layer = nn.Linear(256, num_classes)
      self.output_layer = nn.LogSoftmax(dim=1)

  def forward(self, x):
      z = self.last_layer(self.compute_features(x))
      y_pred = F.log_softmax(z, dim=1)

      return y_pred


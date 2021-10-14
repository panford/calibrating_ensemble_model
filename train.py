import torch
import torch.nn as nn
import torch.nn.functional as F
from get_mnist import MNIST
from models import SoftmaxModel as smodel
from tqdm import tqdm

epochs = 20
lr = 0.05
ensemble_size = 5
dataset = "FashionMNIST"
seed = 100

loss_fn = F.nll_loss

input_size, num_classes, train_dataset, test_dataset = MNIST()
kwargs = {"num_workers": 4, "pin_memory": True}

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, **kwargs
)

milestones = [10, 20]
ensemble = [smodel(input_size, num_classes).cuda() for _ in range(ensemble_size)]

def train(model, train_loader, optimizer, epoch, loss_fn):
    model.train()

    total_loss = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = loss_fn(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = torch.tensor(total_loss).mean()
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


optimizers = []
schedulers = []

for model in ensemble:
    # Need different optimisers to apply weight decay and momentum properly
    # when only optimising one element of the ensemble
    optimizers.append(
        torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
    )

    schedulers.append(
        torch.optim.lr_scheduler.MultiStepLR(
            optimizers[-1], milestones=milestones, gamma=0.1
        )
    )

for epoch in range(1, epochs + 1):
    for i, model in enumerate(ensemble):
        train(model, train_loader, optimizers[i], epoch, loss_fn)
        schedulers[i].step()

    # test(ensemble, test_loader, loss_fn)

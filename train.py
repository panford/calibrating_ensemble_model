import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_data import MNIST
from models import Model as base_model
from tqdm import tqdm
from absl import app, flags
from utils import save_checkpoint

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 20, 'num of epochs', lower_bound=1)
flags.DEFINE_integer('num_ensemble', 5, 'number of ensemble models')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_integer('seed', 100, 'random seed')
flags.DEFINE_float('momentum', 0.9, 'momentum for optimizer')
flags.DEFINE_float('weight_decay', 5e-4, 'optimizer weight decay')
flags.DEFINE_string('chkpt_path', './checkpoints', "checkpointing path")


def train(model, train_loader, optimizer, epoch, criterion):
    model.train()

    total_loss = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = criterion(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = torch.tensor(total_loss).mean()
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")

def main(argv):

  criterion = F.nll_loss

  train_dataset, _ , input_dim, num_classes = MNIST()
  kwargs = {"num_workers": 4, "pin_memory": True}

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=FLAGS.batch_size, shuffle=True, **kwargs
  )

  milestones = [10, 20]
  ensemble = [base_model(input_dim, num_classes).cuda() for _ in range(FLAGS.num_ensemble)]

  optimizers = []
  schedulers = []

  for model in ensemble:
      optimizers.append(
          torch.optim.SGD(
              model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay
          )
      )

      schedulers.append(
          torch.optim.lr_scheduler.MultiStepLR(
              optimizers[-1], milestones=milestones, gamma=0.1
          )
      )

  for epoch in range(1, FLAGS.epochs + 1):
      for i, model in enumerate(ensemble):
          train(model, train_loader, optimizers[i], epoch, criterion)
          schedulers[i].step()

  kwargs = {}
  for i, model in enumerate(ensemble):
    model_state_dict_i = 'model'+str(i)+'state_dict'
    optim_state_dict_i = 'optim'+str(i)+'state_dict'

    kwargs[model_state_dict_i] = model.state_dict()
    kwargs[optim_state_dict_i] = optimizers[i].state_dict()
      # test(ensemble, test_loader, criterion)
  kwargs['num_ensemble'] = FLAGS.num_ensemble
  kwargs['model_args'] = {'input_dim':input_dim, 'num_classes':num_classes}
  save_checkpoint(FLAGS.chkpt_path, kwargs)

if __name__ == '__main__':
  app.run(main)
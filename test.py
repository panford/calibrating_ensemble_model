import torch
from utils import load_checkpoint
from models import Model as base_model
import torch.nn.functional as F
from absl import app, flags
from prepare_data import MNIST

FLAGS = flags.FLAGS

flags.DEFINE_string('chkpt_path','./checkpoints', "checkpoint path")
flags.DEFINE_bool('return_pred_targ', 'False', 'boolean return predictions and targets')

def test(models, test_loader, criterion, return_pred_targ):
  for model in models:
    model.eval()

  loss = 0
  correct = 0
  pred_targ_dict = {'pred':[], 'targ':[]}
  for data, target in test_loader:
    with torch.no_grad():
      data = data.cuda()
      target = target.cuda()

      losses = torch.empty(len(models), data.shape[0])
      predictions = []
      for i, model in enumerate(models):
          predictions.append(model(data))
          losses[i, :] = criterion(predictions[i], target, reduction="sum")

      predictions = torch.stack(predictions)

      loss += torch.mean(losses)
      avg_prediction = predictions.exp().mean(0)

      # get the index of the max log-probability
      class_prediction = avg_prediction.max(1)[1]
      correct += (
          class_prediction.eq(target.view_as(class_prediction)).sum().item()
      )
      if FLAGS.return_pred_targ:
        pred_targ_dict['pred'].extend(class_prediction)
        pred_targ_dict['targ'].extend(target)

  loss /= len(test_loader.dataset)

  percentage_correct = 100.0 * correct / len(test_loader.dataset)

  print(
      "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
          loss, correct, len(test_loader.dataset), percentage_correct
      )
  )

  if FLAGS.return_pred_targ:
    return loss, percentage_correct, pred_targ_dict
  else: 
    return loss, percentage_correct



def main(argv):
  checkpoint = load_checkpoint(FLAGS.chkpt_path)
  _, test_dataset, input_dim, num_classes = MNIST()
  criterion = F.nll_loss
  ensemble = [base_model(input_dim, num_classes).cuda() for _ in range(checkpoint['num_ensemble'])]

  for i, model in enumerate(ensemble):
    model.load_state_dict(checkpoint['model'+str(i)+'state_dict'])
  kwargs = {"num_workers": 4, "pin_memory": True}

  test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=5000, shuffle=False, **kwargs
)
  print('test loader size: ', len(test_loader.dataset))
  return test(ensemble, test_loader, criterion, FLAGS.return_pred_targ)

if __name__ == '__main__':
  app.run(main)





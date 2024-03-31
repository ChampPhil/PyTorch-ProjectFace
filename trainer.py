import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


train_data = datasets.FashionMNIST(
    root='train', #Download data to directory named 'train'
    train=True, #Use the TRAINING data
    download=True, #Download the dataset, if its already downloaded, this step won't happen
    transform=transforms.ToTensor() #convert PIL Image or Numpy array to Tensor to shape (C, H, W) in the range 0-1.0
)

test_data = datasets.FashionMNIST(
    root='test',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

class_names = train_data.classes
class_to_idx = train_data.class_to_idx

#DataLoader iterable for training data
train_dataloader = DataLoader(
    dataset = train_data,
    batch_size = 32,
    shuffle = True
)

#DataLoader iterable for test data
test_dataloader = DataLoader(
    dataset = test_data,
    batch_size = 32
)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               acc_fn,
               f1score_fn,
               device):
  train_loss = 0 #For every batch, compute loss for that batch.
  #Then, for every epoch, divide total loss by number of batches to get the average loss per epoch
  #sum of values/number of values = average

  train_accuracy, train_f1_score = 0, 0
  model.train()

  for batch, (X, y) in enumerate(data_loader):
    X, y = X.to(device), y.to(device)
    #Forward pass
    y_logits = model(X)

    #Calculate loss
    loss = loss_fn(y_logits, y)
    train_loss += loss #Acculumate Train loss

    #Compute metrics for this batch
    train_accuracy += acc_fn(torch.argmax(torch.softmax(y_logits, dim=1), dim=1).int(), y)
    train_f1_score += f1score_fn(torch.argmax(torch.softmax(y_logits, dim=1), dim=1).int(), y)

    #Zero the gradients
    optimizer.zero_grad()

    #Perform backpropogration (compute/store the gradients )
    loss.backward()

    #Perform gradient descent by using the negative gradients * step size
    optimizer.step()

  train_loss /= len(data_loader)
  train_accuracy /= len(data_loader)
  train_f1_score /= len(data_loader)

  print(f"Training Loss: {train_loss:.2f} | Training Accuracy: {train_accuracy:.2f} | Training F1Score: {train_f1_score:.2f}")


def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn,
               f1score_fn,
               device):

  test_loss, test_accuracy, test_f1_score = 0, 0, 0
  model.eval()
  with torch.inference_mode():
    for test_X, test_y in data_loader:
      test_X, test_y = test_X.to(device), test_y.to(device)
      #Forward Pass
      test_logits = model(test_X)

      test_loss += loss_fn(test_logits, test_y)

      #Compute metrics
      test_accuracy += acc_fn(torch.argmax(torch.softmax(test_logits, dim=1), dim=1).int(), test_y)
      test_f1_score += f1score_fn(torch.argmax(torch.softmax(test_logits, dim=1), dim=1).int(), test_y)

    test_loss /= len(data_loader)
    test_accuracy /= len(data_loader)
    test_f1_score /= len(data_loader)

  print(f"Testing Loss: {test_loss:.2f} |  Test Accuracy: {test_accuracy:.2f}  | Test F1Score: {test_f1_score:.2f}")




class FashionMNISTV1(nn.Module):
  def __init__(self, hidden_units, in_features, out_features):
    super().__init__()
    self.stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=in_features, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=out_features)

    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.stack(x)

torch.manual_seed(42)
model1 = FashionMNISTV1(
    hidden_units=64,
    in_features=28*28,
    out_features=len(class_names)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model1.parameters(), lr=0.1)







from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device = None):
  total_time = end - start
  print(f"Total time on {device}: {total_time:.3f} seconds")
  return total_time




torch.manual_seed(42)

from torchmetrics import Accuracy, F1Score
from tqdm.auto import tqdm

acc_fn = Accuracy(task="multiclass", num_classes=len(class_names))
f1_score =  F1Score(task="multiclass", num_classes=len(class_names))

# Measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader,
        model=model1,
        loss_fn=loss_fn,
        optimizer=optimizer,
        acc_fn=acc_fn,
        f1score_fn=f1_score,
        device=device
    )
    test_step(data_loader=test_dataloader,
        model=model1,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        f1score_fn=f1_score,
        device=device
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn,
               f1score_fn,
               device):
  loss, acc, f1score = 0, 0, 0

  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(data_loader):
      X, y = X.to(device), y.to(device)
      y_logits = model(X)

      loss += loss_fn(y_logits, y)
      acc += acc_fn(torch.argmax(torch.softmax(y_logits, dim=1), dim=1).int(), y)
      f1score = f1score_fn(torch.argmax(torch.softmax(y_logits, dim=1), dim=1).int(), y)

    loss /= len(data_loader)
    acc /= len(data_loader)
    f1score /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
              "model_loss": loss.item(),
              "model_acc": acc,
              "model_f1score": f1score}

model_01_results = eval_model(model1, test_dataloader, loss_fn, acc_fn, f1_score, device=device)
print(model_01_results)



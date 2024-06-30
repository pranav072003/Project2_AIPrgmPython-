import numpy as np
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
# import helper
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description = 'Classifier model for images of flowers')
parser.add_argument('--save_dir', type=str, default='/home/workspace/ImageClassifier', help='path to image classifier directory')
parser.add_argument('--dir', type=str, default='/home/workspace/ImageClassifier/flowers', help='path to the dataset f')
parser.add_argument('--arch', type=str, default='vgg13', help='nueral network architecture')
parser.add_argument('--learning_rate', type=int, default=0.00032, help='learning rate for backpropagation')
parser.add_argument('--hidden_units1', type=int, default=512, help='hidden layer nodes for 1st hidden layer')
parser.add_argument('--hidden_units2', type=int, default=128, help='hidden layer nodes for 2nd hidden layer')
parser.add_argument('--epochs', type=int, default=4, help='no. of epochs')
parser.add_argument('--gpu', type=bool, default=False, help='enables GPU')
parse = parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms =  transforms.Compose([transforms.CenterCrop(224),
                                      transforms.Resize(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                         [0.229,0.224,0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(data_dir, transform = data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = 24, shuffle = True)

img_train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
img_valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
img_test_datasets = datasets.ImageFolder(test_dir, transform = data_transforms)

trainloader = torch.utils.data.DataLoader(img_train_datasets, batch_size = 24, shuffle = True)   

validloader = torch.utils.data.DataLoader(img_valid_datasets, batch_size = 24, shuffle = True)   

testloader = torch.utils.data.DataLoader(img_test_datasets, batch_size = 24, shuffle = True) 

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    
import signal

from contextlib import contextmanager

import requests


DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}


def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable
 
arch = parse.arch

if arch=='vgg13':
    model = models.vgg13(pretrained=True)
    inp = 25088
    model.classifier = nn.Sequential(nn.Linear(inp,parse.hidden_units1),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(parse.hidden_units1,       parse.hidden_units2),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(parse.hidden_units2, 102),
                                 nn.LogSoftmax(dim = 1)
                                 )
elif arch=='densenet121':
    model = models.densenet121(pretrained=True)
    inp = 1024
    model.classifier = nn.Sequential(nn.Linear(inp,parse.hidden_units1),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(parse.hidden_units1,       parse.hidden_units2),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(parse.hidden_units2, 102),
                                 nn.LogSoftmax(dim = 1)
                                 )
else:
    model = models.vgg13(pretrained=True)
    inp = 25088
    model.classifier = nn.Sequential(nn.Linear(inp,parse.hidden_units1),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(parse.hidden_units1,       parse.hidden_units2),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(parse.hidden_units2, 102),
                                 nn.LogSoftmax(dim = 1)
                                 )

avail = parse.gpu
device = torch.device("cuda" if avail else "cpu")

for param in model.parameters():
    param.requires_grad = False
    
for param in model.classifier.parameters():
    param.requires_grad=True

with active_session():
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= parse.learning_rate)
    epochs = parse.epochs
    
    model.to(device)
    step = 0
    
    for e in range(epochs):
        for images, labels in trainloader:
            running_loss = 0
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            step += 1
            
            running_loss += loss.item()
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Training loss: {running_loss/len(trainloader):.3f}")
    print(step)
    
with active_session():
    model.eval()
    device = torch.device("cuda" if avail else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            test_loss = 0
            accuracy = 0
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            
            print('Test loss: {}, Accuracy: {}'.format(test_loss/len(trainloader),accuracy))
        model.train()

with active_session():
    model.class_to_idx = img_train_datasets.class_to_idx

    checkpoint = {'input_size': inp,
                  'arch': arch,
              'output_size': 102,
              'hidden_layers': [parse.hidden_units1, parse.hidden_units2],
              'state_dict': model.state_dict(),
               'epochs': epochs,
              'learning_rate': parse.learning_rate,
               'optimizer_state': optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
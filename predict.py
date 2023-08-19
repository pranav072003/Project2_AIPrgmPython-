import numpy as np
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser(description = 'Testing  flower image classifier')
parser.add_argument('--img_dir', type=str, default='/home/workspace/ImageClassifier/flowers/valid/10/image_07102.jpg', help='path to the image to be tested')
parser.add_argument('--check', type=str, default='checkpoint.pth', help='path to checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='no. of most probable classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to class to index mapped file')
parser.add_argument('--gpu', type=bool, default=False, help='enables gpu')

in_arg = parser.parse_args()

avail = in_arg.gpu
img = in_arg.img_dir
checkfile = in_arg.check
topk = in_arg.top_k
cat_to_name = in_arg.category_names

import json

with open(cat_to_name, 'r') as f:
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


def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    if checkpoint['arch']=='vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'],checkpoint['hidden_layers'][0]),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(checkpoint['hidden_layers'][0],       checkpoint['hidden_layers'][1]),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(checkpoint['hidden_layers'][1], checkpoint['output_size']),
                                 nn.LogSoftmax(dim = 1)
                                 )
#     optimizer = optim.Adam(model.parameters(), lr= checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return model

model = load_checkpoint(checkfile)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
    # TODO: Process a PIL image for use in a PyTorch model    
    im = image
    im = im.resize((256, 256))
    
    #referred the below given code snippet from GFG(Geeks for Geeks)
    #article regarding modifying PIL Images
    width, height = im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im = im.crop((left, top, right, bottom))
    #referred only till this point
    
    np_image = np.array(im)
    
    mean_arr = np.array([0.485, 0.456, 0.406])
    std_arr = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean_arr) / std_arr   
    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if avail else "cpu")
    model.to('cpu')
    model = model.double()
    
    # TODO: Implement the code to predict the class from an image file
    data_transforms =  transforms.Compose([transforms.CenterCrop(224),
                                      transforms.Resize(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    image = Image.open(image_path)
    image = process_image(image)
    image1 = torch.from_numpy(image)
    image1.unsqueeze_(0)
    log_ps = model(image1)
    ps = torch.exp(log_ps)
    probs, classes = ps.topk(topk, dim = 1)
    return probs[0], classes[0]

probs, classes = predict(img, model, topk)
classes1 = classes.numpy()
probs1 = probs.detach().numpy()

classes2 = [cat_to_name[str(i)] for i in classes1]
print('Probabilities of ',topk,' classes are:-')
print(probs1)
print('Classes are:-')
print(classes2)

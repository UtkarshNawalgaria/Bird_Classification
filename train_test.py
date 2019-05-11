import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from utils import *

if gpu:
    vgg16.cuda()

###################################################################################################
                                          #TRAINING#
###################################################################################################

# load the training set into the dataloader.
trn_set = BirdDataset('.', True, tnfs['train'])
trainloader = DataLoader(dataset=trn_set,
                         shuffle=True,
                         batch_size=batch_size,
                         num_workers=num_workers)

# defining the model
# using vgg16 as the feature extractor for this dataset.
vgg16 = models.vgg16(pretrained=True)

# freeze training for all the "features" layers.
for param in vgg16.features.parameters():
    param.requires_grad=False

# add the new last layer for the model
inputs = vgg16.classifier[6].in_features
last_layer = nn.Linear(inputs, 200)
vgg16.classifier[6] = last_layer

# define the loss function and the optimization strategy.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=lr)

# training begins.
for epoch in range(1, epochs+1):

    # calculate the training loss.
    train_loss = 0.0

    # get batches of image to train the classifier.
    for batch, (images, target) in enumerate(trainloader):

        if gpu:
            images, target = images.cuda(), target.cuda()

        # initialize the optimizer parameters with zero.
        optimizer.zero_grad()
        # put the images through the model.
        output = vgg16(images)
        # compute the loss
        loss = criterion(output, target)
        # do backpropagation
        loss.backward()
        # update the parameters.
        optimizer.step()
        # calculate the training loss.
        train_loss += loss.item()

        # print the loss every 20 batches.
        if batch % 20 == 19:
            print(f"Epoch {epoch}, Batch {batch+1}, loss {train_loss/32}")
            train_loss = 0.0

###################################################################################################
                                          #TESTING#
###################################################################################################

# load the testing set into testloader
test_set = BirdDataset('.', False, tnfs['test'])
testloader = DataLoader(dataset=test_set,
                        batch_size=batch_size,
                        num_workers=num_workers)

# set the model into evaluation mode.
vgg16.eval()
# defining the test loss.
test_loss = 0.0
# start the inference process.
for data, target in testloader:

  if gpu:
    data, target = data.cuda(), target.cuda()

  output = vgg16(data)

  _, pred = torch.max(output.data, 1)
  total += labels.size(0)
  correct += (pred == labels).sum().item()

print(f"Accuracy of the network: {100*correct/total}")



# File Nature:          HKUST; COMP5212 Fall 2022; Programming Homework 1
# Author:               LIANG, Yuchen
# SID:                  20582717
# Last edited date:     12 OCT 2022

# Don't change batch size
batch_size = 64

# Don't change unless necessary
input_size = 28*28
num_classes = 1

# Init training parameters
num_epochs = 5
learning_rate = 2
momentum = 0

# Imports
import sys
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from tkinter import Label
# from turtle import delay

print("Current Python version: ", sys.version)
print("Current Pytorch version: ", torch.__version__)

## USE THIS SNIPPET TO GET BINARY TRAIN / TEST DATA

train_data = datasets.MNIST('/data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('/data/mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

subset_indices = ((train_data.targets == 0) + (train_data.targets == 1)).nonzero()
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))

subset_indices = ((test_data.targets == 0) + (test_data.targets == 1)).nonzero()
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))

# print(train_data.train_data.size())
# print(train_data.targets.size())

# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.

class SVM(nn.modules.Module):

    def __init__(self):
        super(SVM,self).__init__()

    def forward(self, outputs, labels):
         return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size

svm_model = nn.Linear(input_size,num_classes,bias=False)
svm_model.requires_grad_()

ciriterion = SVM()
optimizer = torch.optim.SGD(svm_model.parameters(), lr=learning_rate, momentum=momentum)

print(" ")
print("__________________________Start Training________________________")
total_step = 0
for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)
        labels = Variable(2*(labels.float()-0.5))

        # Forward pass
        outputs = svm_model(images)
        loss_svm = ciriterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss_svm.backward()
        optimizer.step()

        total_batches += 1
        batch_loss += loss_svm.item()
        avg_loss_epoch = batch_loss/total_batches

        total_step += 1

    print ('Total step {}, Epoch {}/{}, Average Loss: {:.4f}'.format(total_step, epoch+1, num_epochs, avg_loss_epoch ))

print(" ")
print("______________________________Testing______________________________")
# Test the SVM Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = images.reshape(-1, 28*28)

    outputs = svm_model(images)
    predicted = outputs.data >= 0
    total += labels.size(0)
    correct += (predicted.view(-1).long() == labels).sum()
    accuracy = 100 * (correct.float() / total)
print('correct: {}. total: {}. accuracy: {} %'.format(correct, total, accuracy))

print(" ")
print("____________________________RESULT______________________________")
print('Accuracy of the model on the test images: %f %%' % (accuracy))
print("number of total test images", total)
print("numbers of correctly predicted test images" ,correct )
print(" ")
print("Numbers of epochs:", num_epochs)
print("learning rate    :", learning_rate)
print("Momentum         :", momentum)
print(" ")
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
num_epochs = 10
learning_rate = 0.0001
momentum = 0.4

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

print(" ")
print("__________________________Version Check__________________________")
print("Please compare the versions below with the documentation if needed")
print("Current Python version: ", sys.version)
print("Current Pytorch version: ", torch.__version__)
# print("Current Matplotlib version", matplotlib.__version__)

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
subset_indices = ((train_data.targets == 0) + (train_data.targets == 1)).nonzero().reshape(-1)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))


subset_indices = ((test_data.targets == 0) + (test_data.targets == 1)).nonzero().reshape(-1)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))

# print(train_data.train_data.size())
# print(train_data.targets.size())

# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.

class logistic_regression(nn.modules.Module):

    def __init__(self):
        super(logistic_regression,self).__init__()

    def forward(self, outputs, labels):
        batch_size = outputs.size()[0]
        return torch.sum(torch.log(1 + torch.exp(-(outputs.t()*labels))))/batch_size

model = nn.Linear(input_size,num_classes,bias=False)
model.requires_grad_()

criterion = logistic_regression()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum= momentum)

print(" ")
print("__________________________Start Training__________________________")
total_step = 0
for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)
        labels = Variable(2*(labels.float()-0.5))

        #forward propagation
        outputs = model(images)

        # compute loss based on obtained value and actual label
        loss = criterion(outputs, labels)
        optimizer.zero_grad()

        # backward propagation
        loss.backward()
        optimizer.step()

        total_batches += 1
        batch_loss += loss.item()
        avg_loss_epoch = batch_loss/total_batches

        total_step += 1

    print ('Total step {}, Epoch {}/{}, Average Loss: {:.4f}'.format(total_step, epoch+1, num_epochs, avg_loss_epoch ))


print(" ")
print("_____________________________Testing______________________________")
# Test the logistic Model
correct = 0
total = 0

for images, labels in test_loader:

    reshaped_images = images.reshape(-1, 28*28)

    outputs_test = torch.sigmoid(model(reshaped_images))
    predicted = outputs_test.data >= 0.5

# ## Display image for testing
#     print(images.size(0))
#     display_i = 0
#     display_x = 0
#     display_y = 0
#     fig, axs = plt.subplots(8,8)
#     while(display_i < images.size(0)):
#         while(display_x < 8):
#             while(display_y < 8):
#                 display_img = images[display_i]
#                 display_img = np.array(display_img, dtype='float')
#                 pixels = display_img.reshape((28, 28))
#                 correct_check = predicted.view(-1).long() == labels
#                 axs[display_x,display_y].plot()
#                 axs[display_x,display_y].imshow(pixels, cmap='gray')
#                 axs[display_x,display_y].set_title("Prediction:{}".format(predicted[display_i]),fontsize=10)
#                 # axs[display_x,display_y].set_xlabel()
#                 axs[display_x,display_y].set_ylabel("Image:{},Label:{}".format(display_i+1,labels[display_i]),fontsize=10)
#                 print("Display image and label" , display_i)
#                 print(correct_check)
#                 display_i += 1
#                 display_y += 1
#             display_y = 0
#             display_x += 1
#     manager = plt.get_current_fig_manager()
#     manager.full_screen_toggle()
#     plt.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.4,
#                     hspace=0.5)
#     plt.show(block=False)
#     plt.pause(5)
#     plt.close('all')

    # Total number of labels
    total += labels.size(0)

    # Total correct predictions
    correct += (predicted.view(-1).long() == labels).sum()
    accuracy = 100 * (correct.float() / total)
print('correct: {}. total: {}. accuracy: {} %'.format(correct, total, accuracy))


print(" ")
print("__________________________RESULT____________________________")
print('Accuracy of the model on the test images  : %f %%' % (100 * (correct.float() / total)))
print("Number of total test images               :", total)
print("Number of correctly predicted test images :", correct )
print(" ")
print("Number of epochs :", num_epochs)
print("learning rate    :", learning_rate)
print("Momentum         :", momentum)
print(" ")
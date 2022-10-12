<!--
# File Nature:          HKUST; COMP5212 Fall 2022; Programming Homework 1 Report
# Author:               LIANG, Yuchen
# SID:                  20582717
# Last edited date:     12 OCT 2022
-->
COMP5212: Machine Learning
Programming Homework 1 Report
=========================================================
<center><b>LIANG, Yuchen Eric 20582717</b></center>
<center><b>yuchen.liang@connect.ust.hk</b></center>
<center><b>Oct 2022</b></center>

- [Programming Homework 1 Report](#programming-homework-1-report)
  - [Code documentation](#code-documentation)
      - [Files](#files)
      - [Prerequisite](#prerequisite)
      - [Parameter settings](#parameter-settings)
      - [How to run](#how-to-run)
  - [Average loss after training each epoch](#average-loss-after-training-each-epoch)
      - [Logistic Regression Model](#logistic-regression-model)
      - [Support-Vector Machine](#support-vector-machine)
  - [Final accuracy of trained model](#final-accuracy-of-trained-model)
      - [Logistic Regression Model](#logistic-regression-model-1)
      - [Support-Vector Machine](#support-vector-machine-1)
  - [Compare result for 2 optimizer](#compare-result-for-2-optimizer)
      - [Logistic Regression Model](#logistic-regression-model-2)
      - [Logistic Regression Model (Momentum)](#logistic-regression-model-momentum)
      - [Compare](#compare)
  - [Effect of different step size (learning rate)](#effect-of-different-step-size-learning-rate)
      - [Support-Vector Machine with step size of 2](#support-vector-machine-with-step-size-of-2)
      - [Support-Vector Machine with step size of 0.1](#support-vector-machine-with-step-size-of-01)
      - [Support-Vector Machine with step size of 0.0001](#support-vector-machine-with-step-size-of-00001)
      - [Effect](#effect)

## Code documentation
#### Files
- HW1_logistic.py:
  This file applies the logistic regression model. This file include the whole process of getting the data needed from the dataset, training the model and testing the model.
- HW1_SVM.py:
  This file applies the Support-Vector Machine. This file include the whole process of getting the data needed from the dataset, training the model and testing the model.

#### Prerequisite
- Python 3.9.0 [MSC v.1927 64 bit (AMD64)]
- Pytorch 1.12.1+cu116
- (If plotting is needed) Matplotlib 3.6.0

#### Parameter settings
- Change the training parameters on the top of the python file:

        # Init training parameters
        num_epochs = 10
        learning_rate = 0.0001
        momentum = 0.9

#### How to run
- Run logistic regression model training and testing

        > cd /directory/path/of/HW1_logistic.py
        > python HW1_logistic.py

- Run SVM model training and testing

        > cd /directory/path/of/HW1_SVM.py
        > python HW1_SVM.py

## Average loss after training each epoch
#### Logistic Regression Model

    Number of epochs : 10
    learning rate    : 0.0001
    Momentum         : 0

    Total step 198, Epoch 1/10, Average Loss: 0.5669
    Total step 396, Epoch 2/10, Average Loss: 0.2951
    Total step 594, Epoch 3/10, Average Loss: 0.1976
    Total step 792, Epoch 4/10, Average Loss: 0.1500
    Total step 990, Epoch 5/10, Average Loss: 0.1220
    Total step 1188, Epoch 6/10, Average Loss: 0.1035
    Total step 1386, Epoch 7/10, Average Loss: 0.0904
    Total step 1584, Epoch 8/10, Average Loss: 0.0806
    Total step 1782, Epoch 9/10, Average Loss: 0.0730
    Total step 1980, Epoch 10/10, Average Loss: 0.0669

#### Support-Vector Machine

    Numbers of epochs: 10
    learning rate    : 0.0001
    Momentum         : 0

    Total step 198, Epoch 1/10, Average Loss: 0.2413
    Total step 396, Epoch 2/10, Average Loss: 0.0346
    Total step 594, Epoch 3/10, Average Loss: 0.0250
    Total step 792, Epoch 4/10, Average Loss: 0.0208
    Total step 990, Epoch 5/10, Average Loss: 0.0183
    Total step 1188, Epoch 6/10, Average Loss: 0.0166
    Total step 1386, Epoch 7/10, Average Loss: 0.0153
    Total step 1584, Epoch 8/10, Average Loss: 0.0143
    Total step 1782, Epoch 9/10, Average Loss: 0.0135
    Total step 1980, Epoch 10/10, Average Loss: 0.0129

## Final accuracy of trained model
#### Logistic Regression Model
    __________________________RESULT____________________________
    Accuracy of the model on the test images  : 99.810875 %
    Number of total test images               : 2115
    Number of correctly predicted test images : tensor(2111)

    Number of epochs : 10
    learning rate    : 0.0001
    Momentum         : 0


#### Support-Vector Machine
    ____________________________RESULT______________________________
    Accuracy of the model on the test images: 99.810875 %
    number of total test images 2115.0
    numbers of correctly predicted test images tensor(2111.)

    Numbers of epochs: 10
    learning rate    : 0.0001
    Momentum         : 0

## Compare result for 2 optimizer
This report will compare the logistic regression model and the logistic regression model with momentum.
#### Logistic Regression Model

    __________________________Start Training__________________________
    Total step 198, Epoch 1/10, Average Loss: 0.4646
    Total step 396, Epoch 2/10, Average Loss: 0.2589
    Total step 594, Epoch 3/10, Average Loss: 0.1797
    Total step 792, Epoch 4/10, Average Loss: 0.1392
    Total step 990, Epoch 5/10, Average Loss: 0.1148
    Total step 1188, Epoch 6/10, Average Loss: 0.0983
    Total step 1386, Epoch 7/10, Average Loss: 0.0865
    Total step 1584, Epoch 8/10, Average Loss: 0.0776
    Total step 1782, Epoch 9/10, Average Loss: 0.0706
    Total step 1980, Epoch 10/10, Average Loss: 0.0650

    __________________________RESULT____________________________
    Accuracy of the model on the test images  : 99.810875 %
    Number of total test images               : 2115
    Number of correctly predicted test images : tensor(2111)

    Number of epochs : 10
    learning rate    : 0.0001
    Momentum         : 0

#### Logistic Regression Model (Momentum)

    __________________________Start Training__________________________
    Total step 198, Epoch 1/10, Average Loss: 0.1589
    Total step 396, Epoch 2/10, Average Loss: 0.0438
    Total step 594, Epoch 3/10, Average Loss: 0.0306
    Total step 792, Epoch 4/10, Average Loss: 0.0246
    Total step 990, Epoch 5/10, Average Loss: 0.0210
    Total step 1188, Epoch 6/10, Average Loss: 0.0186
    Total step 1386, Epoch 7/10, Average Loss: 0.0169
    Total step 1584, Epoch 8/10, Average Loss: 0.0155
    Total step 1782, Epoch 9/10, Average Loss: 0.0145
    Total step 1980, Epoch 10/10, Average Loss: 0.0136

    __________________________RESULT____________________________
    Accuracy of the model on the test images  : 99.858162 %
    Number of total test images               : 2115
    Number of correctly predicted test images : tensor(2112)

    Number of epochs : 10
    learning rate    : 0.0001
    Momentum         : 0.9

#### Compare
![OPT](/HW1/pic/OPT.png)
From the above two result we can see that with momentum involved, the loss start at the a smaller initial value and converge more quickly. It can be observed that with momentum involved, the optimization process is accelerated.

## Effect of different step size (learning rate)
This report will use the Support-Vector Machine with different step size to illustrate the effect of step size
#### Support-Vector Machine with step size of 2
    __________________________Start Training________________________
    Total step 198, Epoch 1/5, Average Loss: 0.1249
    Total step 396, Epoch 2/5, Average Loss: 0.0461
    Total step 594, Epoch 3/5, Average Loss: 0.0243
    Total step 792, Epoch 4/5, Average Loss: 0.0167
    Total step 990, Epoch 5/5, Average Loss: 0.0125

    ____________________________RESULT______________________________
    Accuracy of the model on the test images: 99.905434 %
    number of total test images 2115.0
    numbers of correctly predicted test images tensor(2113.)

    Numbers of epochs: 5
    learning rate    : 2
    Momentum         : 0
#### Support-Vector Machine with step size of 0.1
    __________________________Start Training________________________
    Total step 198, Epoch 1/5, Average Loss: 0.0140
    Total step 396, Epoch 2/5, Average Loss: 0.0045
    Total step 594, Epoch 3/5, Average Loss: 0.0031
    Total step 792, Epoch 4/5, Average Loss: 0.0019
    Total step 990, Epoch 5/5, Average Loss: 0.0014

    ____________________________RESULT______________________________
    Accuracy of the model on the test images: 99.952721 %
    number of total test images 2115.0
    numbers of correctly predicted test images tensor(2114.)

    Numbers of epochs: 5
    learning rate    : 0.1
    Momentum         : 0
#### Support-Vector Machine with step size of 0.0001
    __________________________Start Training________________________
    Total step 198, Epoch 1/5, Average Loss: 0.3264
    Total step 396, Epoch 2/5, Average Loss: 0.0522
    Total step 594, Epoch 3/5, Average Loss: 0.0324
    Total step 792, Epoch 4/5, Average Loss: 0.0252
    Total step 990, Epoch 5/5, Average Loss: 0.0214

    ____________________________RESULT______________________________
    Accuracy of the model on the test images: 99.810875 %
    number of total test images 2115.0
    numbers of correctly predicted test images tensor(2111.)

    Numbers of epochs: 5
    learning rate    : 0.0001
    Momentum         : 0

#### Effect
![LR-legend](/HW1/pic/LR-legend.png)
The graph shows that with the decrease of step size, the initial value of the loss decrease first and increase afterwards. This show that the step size need to be at a suitable value (cannot be too large nor too small) in order to reached a optimized situation.
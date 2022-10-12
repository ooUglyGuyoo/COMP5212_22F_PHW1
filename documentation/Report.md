<!--
# File Nature:          HKUST; COMP5212 Fall 2022; Programming Homework 1 Report
# Author:               LIANG, Yuchen
# SID:                  20582717
# Last edited date:     12 OCT 2022
-->
COMP5212: Machine Learning Programming Homework 1 Report
=========================================================
<center><b>LIANG, Yuchen Eric 20582717 | yuchen.liang@connect.ust.hk</b></center>
<center><b>Oct 2022</b></center>

### Average loss after training each epoch
- Logistic Regression Model

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

- Support-Vector Machine

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

### Final accuracy of trained model
- Logistic Regression Model

      Accuracy of the model on the test images  : 99.810875 %
      Number of epochs : 10
      learning rate    : 0.0001
      Momentum         : 0

- Support-Vector Machine

      Accuracy of the model on the test images: 99.810875 %
      Numbers of epochs: 10
      learning rate    : 0.0001
      Momentum         : 0

### Compare result for 2 optimizer
This report will compare the logistic regression model with and without momentum.
<p align="center">
<img src="/HW1/pic/OPT.png" width="350"></img>
</p>
From the above two result we can see that with momentum involved, the loss start at the a smaller initial value and converge more quickly. It can be observed that with momentum involved, the optimization process is accelerated.

### Effect of different step size (learning rate)
This report will use the Support-Vector Machine with different step size to illustrate its effect.
<p align="center">
<img src="/HW1/pic/LR-legend.png" width="350"></img>
</p>
The graph shows that with the decrease of step size, the initial value of the loss decrease first and increase afterwards. This show that the step size need to be at a suitable value (cannot be too large nor too small) in order to reached a optimized situation.
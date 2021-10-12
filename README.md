# EWC-Implementation
Implementation of EWC algorithm proposed in "Overcoming Catastrophic Forgetting in Neural Networks" by Kirkpatrick et al.
main.py can be used for Class Incremental Learning, main_task.py can be used for Task Incremental Learning.
Python Script to Implement: python3.8 main.py [Type of Learning] [# of epochs] [learning rate] [batch size] [sample size] [hidden size] [# of tasks] [importance]
Here:
Type of Learning = 'task_learning' or 'class_learning'
Default values of the rest hyperparameters:
learning rate = 0.001
batch size = 100
sample size = 128
hidden size = 400
No. of tasks = 5
importance = 1000

Reference: [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
Some parts of the code have been inspired by: [EWC.pytorch](https://github.com/moskomule/ewc.pytorch#ewcpytorch)

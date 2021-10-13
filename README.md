# EWC-Implementation
PyTorch Implementation of EWC algorithm proposed in "Overcoming Catastrophic Forgetting in Neural Networks" by Kirkpatrick et al.<br/>
main.py can be used for Class Incremental Learning, main_task.py can be used for Task Incremental Learning.<br/>
Python Script to Implement: python3.8 main.py [Type of Learning] [# of epochs] [learning rate] [batch size] [sample size] [hidden size] [# of tasks] [importance]<br/>
Here:<br/>
Type of Learning = 'task_learning' or 'class_learning'<br/>
Default values of the hyperparameters:<br/>
No. of Epochs = 100<br/>
learning rate = 0.001<br/>
batch size = 100<br/>
sample size = 128<br/>
hidden size = 400<br/>
No. of tasks = 5<br/>
importance = 1000<br/>
<br/>
Reference: [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)<br/>
Some parts of the code have been inspired by: [EWC.pytorch](https://github.com/moskomule/ewc.pytorch#ewcpytorch)<br/>

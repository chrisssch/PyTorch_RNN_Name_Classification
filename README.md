# Classifying Names with a Recurrent Neural Network in PyTorch

Author: Christoph Schauer <br/>
Last updated: 08/05/2019

In the notebook in this repository, I train a simple Recurrent Neural Network (RNN) with a GRU plus a fully connected layer to classify names according to their language using an (unbalanced) dataset consisting of about 20,000 names in 18 languages. After training for 40 epochs, the network achieves an accuracy of around 95%. The notebook also includes code for a standard RNN and a RNN with LSTM as well as a predictor function running on user input.

The code in this notebook is an adaption of the tutorial [Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) by Sean Robertson from the official PyTorch website. I try to follow the naming conventions used there for the most part. The primary changes are:
* Converting the text files containing the names into a quasi-tabular format
* Replacing the custom 1-by-1 data loader with a standard PyTorch batch data loader using sequence padding
* Replacing the custom RNN code with a network using PyTorch's nn.RNN() module plus a fully connected layer
* Updating the predict() function for the new workflow and making it standalone
* Adding classes to build RNNs with GRU and LSTM
* Additional and more informative printouts

I made these changes to bring the workflow more in line with what people learning PyTorch - including myself - will have seen in other basic tutorials and to make it easier to use it as stepping stone for working with other text datasets.

Portions of the neural network code are based on [PyTorch Tutorials: Recurrent Neural Network](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py) by Yunjey Choi.

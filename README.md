Multi-layer perceptron for classification and regression
=============
A python demo code for classification and regression
by Neural Network (multi-layer perceptron) using Chainer
by Shizuo KAJI

## Licence
MIT Licence

# Requirements
- python 3.6

Install [Anaconda](https://www.anaconda.com/download/)
 if you do not have python 3 on your system.

- Chainer 2.0:  `pip install chainer`
- ChainerUI: `pip install chainerui`

# Example
Let us try with the famous [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris).
Download iris.data from the web page.

As MLP.py only accepts numerical data, we have to change label names
to integer values.
`python csv2mlp.py iris.data.txt -l 4 > iris.csv`
does the job. It converts the downloaded file iris.data.txt into a csv compatible with MLP.py

`python MLP.py -h`
gives a brief description of command line arguments

`python MLP.py iris.csv -l 0 -sr 0 -nu 80 80 80 3 -t 5 -a relu -bn`
reads iris.csv and use every 5 entries (-t 5) for validation and the rest for training.
Then, it predicts the value of the 0th column (-l 0, 3 classes)
from other columns using a neural network with 80 80 80 3 units (-nu 80 80 80 3)
and with ReLU as an activation function )-a relu) using Batchnormalization (-bn).
Notice that the number of units in the last layer agrees with the number of the classes.
The result is recorded in the file result/result.csv.

`python MLP.py iris.csv -l 1 -sr 0 -sc 0 -dr 0 -nu 80 80 1 -t 6 --regression`
reads iris.csv and use every 6 entries for validation and the rest for training.
Then, it predicts the value of the 1st column (sepal length)
from columns except for 0th (-sc 0) and 1st.
It plots a graph with matplotlib.
The column indices excluded from training and prediction are specified with -sc
(e.g., -sc 1 3 4 8).

`python MLP.py iris.csv -l 1 2 -sr 0 -dr 0.3 -nu 80 80 2 -t 6 --regression`
does multi-column (-l 1 2) regression at a time.
Notice that the number of target columns should match the number of units in the last layer (in this case 2).


# LIMITATION
CSV files should not contain non-numerical values.
Remove non-numerical entries (except for the header) using csv2mlp.py
(see the above example).

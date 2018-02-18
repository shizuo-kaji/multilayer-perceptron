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

`python MLP.py iris.csv -l 0 -sr 0 -dr 0 -nu 80 -nl 3 -oc 3 -t 5 -a relu`
reads iris.csv and use every 5 entries for validation and the rest for training.
Then, it predicts the value of the 0th column (class)
from other columns using a neural network with 80 units times 5 hidden layers,
and with ReLU as an activation function.
Note that there are three classes in the dataset so -oc 3.
The result is recorded in the file result/nn-out.csv.

`python MLP.py iris.csv -l 0 -sr 0 -dr 0 -nu 80 -nl 3 -oc 3 -t 5 -a relu -g 0`
same as above but with GPU (working installation of cupy with CUDA support is required).

`python MLP.py iris.csv -l 0 -sr 0 -dr 0 -nu 80 -nl 3 -oc 3 --predict -r result/snapshot_iter_1200`
does not perform training but instead read the trained snapshot from result/snapshot_iter_1200
and predicts all entries from iris.csv.

`python MLP.py iris.csv -l 1 -sr 0 -sc 0 -dr 0 -nu 80 -nl 3 -t 6`
reads iris.csv and use every 6 entries for validation and the rest for training.
Then, it predicts the value of the 1st column (sepal length)
from columns except for 0th (specified by -sc) and 1st.
It plots a graph with matplotlib.
The column indices excluded from training and prediction are specified with -sc
(e.g., -sc 1 3 4 8).

# LIMITATION
CSV files should not contain non-numerical values.
Remove non-numerical entries (except for the header) using csv2mlp.py
(see the above example).

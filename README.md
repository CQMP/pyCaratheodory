# pyCaratheodory

A Python package for temperature extrapolation of Matsubara frequency data.

## Installation

1. Clone the repository:
```
git clone git@github.com:CQMP/pyCaratheodory.git
```
2. Install the package:
```
cd pyCaratheodory
pip install -e .
```

## Usage

An example notebook, located at examples/example.ipynb, demonstrates how to reproduce the results presented in Fig. 2 of the paper.

Alternatively, one can use the example script located at examples/example.py to obtain the same results and save them in a text file (example_output.txt) by running the following command:
```
python example.py --inputfile example_input.txt --beta 18 --beta_new 20 --outputfile example_output.txt 
```





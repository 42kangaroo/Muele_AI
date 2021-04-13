# AlphaZero for Mill

This is a school project that aims to learn the game of Mill, also known as Nine Mens Morris, with the AlphaZero
algorithm. It has been parallelized with multiprocessing and uses keras and TensorFlow for the Neural Networks.

![The GUI](GUI_mühle.png)

## Table of Contents

* [How to install](#how-to-install)
* [Usage](#usage)
  * [Training](#training)
  * [Play](#play)
* [Technologies](#technologies)
* [Project Status](#project-status)
* [Documentation](#documentation)
* [Contributors](#contributors)
* [License](#license)
* [Other Methods](#other-methods)

## How to install

To use this project, first clone the repo on your device using the command below:

`git clone -b alphaZero https://github.com/42kangaroo/Muele_AI.git`

Then, change to the directory and install the dependencies:

```
cd Muele_AI
pip install -r requirements.txt
```

## Usage

### Training

Define your own Network in [`Network.py`](Network.py) and change the hyperparameters in [`configs.py`](configs.py). Use
a computer with a GPU for training, or it will need a very long time. To install TensorFlow for the GPU follow these
Tutorials for
[Windows](https://shawnhymel.com/1961/how-to-install-tensorflow-with-gpu-support-on-windows/)
and [Linux](https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d). Then, start the
program with

`py main.py`

The program will return with an error if the files `interrupt_array.npy` or `interrupt_vars.obj`
persist in the main directory. Delete these files in this case.

If you need to stop the program in the middle of training, press `ctrl+C` and wait a little. The
files `interrupt_array.npy` and `interrupt_vars.obj` will appear. If they don't, copy them from the directory you
configured in [`configs.py`](configs.py). The program will then continue at the same point you interrupted it.

The trained net will appear in the configured directory with the name `whole_net`.

### Play

When you open the file [`Graphics.py`](Graphics.py) you will find this line at the end of the file:

```python
MCGraphics = ModeratedGraphics("run5/models/whole_net", 12, 1.05)
MCGraphics.playLoop()
```

Choose the directory of your preferred Network and change it in the file. You can also make it better and slower or
faster and worse by changing the number of simulations.

You can then start the program with

`py Graphics.py`

## Technologies

I used these libraries for my project.

* [Python](https://www.python.org/) 3.7
* [numpy](https://numpy.org/) 1.18.5
* [TensorFlow](https://www.tensorflow.org/) 2.4.1
* [PySimpleGui](https://pysimplegui.readthedocs.io/en/latest/) 4.34.0
* [multiprocessing](https://docs.python.org/3.8/library/multiprocessing.html)
* [ray](https://ray.io/) 2.0.0
* [stellargraph](https://stellargraph.readthedocs.io/en/stable/) 1.2.1

## Project Status

I'm almost finished with this project, and I only need to train it one last time now. I don't have time to train it with
a Graph Neural Network.

## Documentation

You can read the [documentation](https://drive.google.com/file/d/1z9zaC1zZEqTncdVrNIjXunE9AJR4O7gy/view?usp=sharing)
and the [project paper](https://drive.google.com/file/d/1jZlc4MIeE6FR0YXaPvkx2_3wWo1mAWGn/view?usp=sharing) by following
the links (both in german).

## Contributors

Thanks to [@halloalexkern](https://github.com/halloalexkern/) and Oliver Peyron for letting me use there computer for
training. I also want to thank
[@rempfler](https://github.com/rempfler/) for the tips he gave me for the Networks.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE)

## Other Methods

I also tried to learn the game of Mill with MCTS and Reinforcement learning.
See [MCTS](https://github.com/42kangaroo/Muele_AI/tree/mcts)
and [Reinforce](https://github.com/42kangaroo/Muele_AI/tree/reinforce) branch.
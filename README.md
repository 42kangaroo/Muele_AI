# Reinforcement Learning for Mill

This is a school project that aims to learn the game of Mill, also known as Nine Mens Morris, with the Reinforcement
Learning algorithm. It uses keras and TensorFlow for the Neural Networks.

![The GUI](GUI_mühle.png)

## Table of Contents

* [How to install](#how-to-install)
* [Usage](#usage)
    * [Training](#training)
    * [Play](#play)
* [Technologies](#technologies)
* [Project Status](#project-status)
* [Documentation](#documentation)
* [License](#license)
* [Other Methods](#other-methods)

## How to install

To use this project, first clone the repo on your device using the command below:

`git clone -b reinforce https://github.com/42kangaroo/Muele_AI.git`

Then, change to the directory and install the dependencies:

```
cd Muele_AI
pip install -r requirements.txt
```

## Usage

### Training

Define your own network in the `Model` section of [`Mill.ipynb`](Mill.ipynb) and change the hyperparameters in the
Controller section at the bottom.

```python
cont = Controler(0.05,0.4,50000,0.7,3,60000,0.4,1,60000, 0.5, 32,0.1,24,100000,f"Move0-{dt.datetime.now().strftime('%d%m%Y%H%M')}", 256,100,0)
cont.train(2000)
```

Then, run this section. The trained Network will then be saved to the `model/` directory.

### Play

To play against yourself or a random agent, run the `Moderated Play` section of [`Mill.ipynb`](Mill.ipynb). It is
currently not possible to play against a trained network.

## Technologies

I used these libraries for my project.

* [Python](https://www.python.org/) 3.7
* [numpy](https://numpy.org/) 1.18.5
* [TensorFlow](https://www.tensorflow.org/) 2.4.1
* [PySimpleGui](https://pysimplegui.readthedocs.io/en/latest/) 4.34.0
* [scikit learn](https://scikit-learn.org/stable/index.html) 0.24.1

## Project Status

It is not possible to learn the game of Mill with the Reinforcement Learning strategies as it doesn't even learn valid
moves. So I implemented MCTS and AlphaZero. See [Other Methods](#other-methods)

## Documentation

You can read the [documentation](https://drive.google.com/file/d/1z9zaC1zZEqTncdVrNIjXunE9AJR4O7gy/view?usp=sharing)
and the [project paper](https://drive.google.com/file/d/1jZlc4MIeE6FR0YXaPvkx2_3wWo1mAWGn/view?usp=sharing) by following
the links.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE)

## Other Methods

I also tried to learn the game of Mill with MCTS and AlphaZero.
See [MCTS](https://github.com/42kangaroo/Muele_AI/tree/mcts)
and [AlphaZero](https://github.com/42kangaroo/Muele_AI/tree/alphaZero) branch.
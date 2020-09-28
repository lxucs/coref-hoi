# End-to-End Coreference Resolution with Different Higher-Order Inference Methods

This repository contains the implementation of the paper: [Revealing the Myth of Higher-Order Inference in Coreference Resolution](https://arxiv.org/abs/2009.12013).

## Architecture

The basic end-to-end coreference model is a PyTorch re-implementation based on the TensorFlow model, and it follows the same preprocessing (see this [repository](https://github.com/mandarjoshi90/coref)).

There are four higher-order inference (HOI) methods experimented: **Attended Antecedent**, **Entity Equalization**, **Span Clustering**, and **Cluster Merging**. All are included here except for Entity Equalization which is experimented in the equivalent TensorFlow environment (see this separate [repository](https://github.com/lxucs/coref-ee)).

**File Structure**:
* [run.py](run.py): training and evaluation
* [model.py](model.py): the coreference model
* [higher_order.py](higher_order.py): higher-order inference modules
* [analyze.py](analyze.py): result analysis
* [preprocess.py](preprocess.py): converting CoNLL files to examples
* [tensorize.py](tensorize.py): tensorizing example
* [conll.py](conll.py), [metrics.py](metrics.py): same CoNLL-related files from the [repository](https://github.com/mandarjoshi90/coref)
* [experiments.conf](experiments.conf): different model configurations

## Basic Setup


## Evaluation
Trained models are provided [here](https://drive.google.com/drive/folders/1MqW38YGYJtLHW8i20LPRBPt47CoOLuyW?usp=sharing). Each configuration has its own directory with two trained models included inside.

1. 

## Training
TBD soon.




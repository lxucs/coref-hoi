# End-to-End Coreference Resolution with Different Higher-Order Inference Methods

This repository contains the implementation of the paper: Revealing the Myth of Higher-Order Inference in Coreference Resolution.

## Architecture

The basic end-to-end coreference model is a PyTorch re-implementation based on the TensorFlow model, and it follows the same preprocessing (see this [repository](https://github.com/mandarjoshi90/coref)).

There are four higher-order inference (HOI) methods experimented: Attended Antecedent, Entity Equalization, Span Clustering, and Cluster Merging. All are included here except for Entity Equalization which is experimented in the equivalent TensorFlow environment (see this separate [repository](https://github.com/lxucs/coref-ee)).

**File Structure**:
* [run.py](run.py)


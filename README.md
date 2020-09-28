# End-to-End Coreference Resolution with Different Higher-Order Inference Methods

This repository contains the implementation of the paper: [Revealing the Myth of Higher-Order Inference in Coreference Resolution](https://arxiv.org/abs/2009.12013).

## Architecture

The basic end-to-end coreference model is a PyTorch re-implementation based on the TensorFlow model following similar preprocessing (see this [repository](https://github.com/mandarjoshi90/coref)).

There are four higher-order inference (HOI) methods experimented: **Attended Antecedent**, **Entity Equalization**, **Span Clustering**, and **Cluster Merging**. All are included here except for Entity Equalization which is experimented in the equivalent TensorFlow environment (see this separate [repository](https://github.com/lxucs/coref-ee)).

**Files**:
* [run.py](run.py): training and evaluation
* [model.py](model.py): the coreference model
* [higher_order.py](higher_order.py): higher-order inference modules
* [analyze.py](analyze.py): result analysis
* [preprocess.py](preprocess.py): converting CoNLL files to examples
* [tensorize.py](tensorize.py): tensorizing example
* [conll.py](conll.py), [metrics.py](metrics.py): same CoNLL-related files from the [repository](https://github.com/mandarjoshi90/coref)
* [experiments.conf](experiments.conf): different model configurations

## Basic Setup
Set up environment and data for training and evaluation:
* Install Python3 dependencies: `pip install -r requirements.txt`
* Create a directory for data that will contain all data files, models and log files; set `data_dir = /path/to/data/dir` in [experiments.conf](experiments.conf)
* Prepare dataset (requiring [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) corpus): `./setup_data.sh /path/to/ontonotes /path/to/data/dir`

For SpanBERT, download the pretrained weights from this [repository](https://github.com/facebookresearch/SpanBERT), and rename it `/path/to/data/dir/spanbert_base` or `/path/to/data/dir/spanbert_large` accordingly.

## Evaluation
Provided trained models:
* SpanBERT + no HOI: [download](https://cs.emory.edu/~lxu85/train_spanbert_large_ml0_d1.tar)
* SpanBERT + Attended Antecedent: [download](https://cs.emory.edu/~lxu85/train_spanbert_large_ml0_d2.tar)
* SpanBERT + Span Clustering: [download](https://cs.emory.edu/~lxu85/train_spanbert_large_ml0_sc.tar)
* SpanBERT + Cluster Merging: [download](https://cs.emory.edu/~lxu85/train_spanbert_large_ml0_cm_fn1000_max_dloss.tar)
* SpanBERT + Entity Equalization: see [repository](https://github.com/lxucs/coref-ee)

The name of each directory corresponds with a **configuration** in [experiments.conf](experiments.conf). Each directory has two trained models inside.

Evaluate a model on the dev/test set:
* Download the corresponding model directory and put it under `data_dir`
* `python evaluate.py [config] [model_id] [gpu_id]`
    * e.g. Attended Antecedent:`python evaluate.py train_spanbert_large_ml0_d2 May08_12-38-29_58000 0`

## Training
`python run.py [config] [gpu_id]`

* [config] can be any **configuration** in [experiments.conf](experiments.conf)
* Log file will be saved at `your_data_dir/[config]/log_XXX.txt`
* Models will be saved at `your_data_dir/[config]/model_XXX.bin`
* Tensorboard is available at `your_data_dir/tensorboard`


## Configurations
Some important configurations in [experiments.conf](experiments.conf):
* `data_dir`: the full path to the directory containing dataset, models, log files
* `coref_depth` and `higher_order`: controlling the higher-order inference module
* `bert_pretrained_name_or_path`: the name/path of the pretrained BERT model ([HuggingFace BERT models](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained))
* `max_training_sentences`: the maximum segments to use when document is too long; for BERT-Large and SpanBERT-Large, set to `3` for 32GB GPU or `2` for 24GB GPU

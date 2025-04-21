# Legalese: Replacing prescriptive prepositions with plain language for readability in legal English

This project focuses on the use of complex prepositions in legal English, and aims to take the first step in translating legal --> colloquial English by replacing these wordy complex prepositions with simpler ones.

## Description

This script firstly processes the Cambridge Law Corpus (CLC) Mini, getting information about the sentence and word counts in its contents, and then finds complex prepositions within the corpus. It ranks these found complex prepositions by most common to least common, creates a phrase map of the most common and their simple alternatives, and then trains a Flan-T5 model to replace the complex prepositions with the simple ones. 

## Getting Started

### Dependencies

This script was written on a MacBook Air M2 running Sonoma 14.6.1, and originally in a Jupyter notebook where the kernel environment was running Python 3.11.11.

To use, the following libraries need to be installed and imported:
* import nltk, re, pprint
* from nltk import word_tokenize, sent_tokenize
* import os
* import json
* import xml.etree.ElementTree as ET
* from nltk.corpus.reader.api import CorpusReader
* from nltk.corpus.reader.util import find_corpus_fileids
* import spacy
* from itertools import islice
* import csv
* from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
* from datasets import load_dataset
* import torch

Depending on your corpus, you may not need to unpack it the way I did with json, xml ElementTree, or csv. This corpus just needed a lot of preprocessing.

### Installing

* Please download the .ipynb file to run the Jupyter notebook. It was not tested in pure .py form.
* If you are not running on an Apple Silicon machine, the lines steering the model away from FP16 (half-precision floating point) and onto the MPS are likely unnecessary or detrimental. Depending on your machine, let the model use the CPU to train, or double-check in the T5 documentation which memory it should use.

## Help

If you are on an Apple Silicon machine in the same environment as I was, be sure to run this in your terminal before training:
```
accelerate config
```
and choose the given answers to the following questions:
* In which compute environment are you running?: This machine
* Which type of machine are you using?: No distributed training
* Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?: [yes/NO]:NO
* Do you wish to optimize your script with torch dynamo?: [yes/NO]:no
* Do you wish to use mixed precision?: no 

## Authors

Anna DeLotto
email: delottoa@tcd.ie

## Acknowledgments

Thank you to the University of Cambridge for providing a mini, open-source version of the illustrious Cambridge Law Corpus.

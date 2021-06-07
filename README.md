# Unfair_tos
This project tries to find potentially unfair clauses in terms of service (TOS).  This task categorises a dataset comprising a binary classification task: Positive labels are potentially unfair contractual terms (clauses) from the terms of service. 

A contractual term is unfair if: 
it has not been individually negotiated contrary to the requirement of good faith, 
it causes a significant imbalance in the parties rights and obligations, to the detriment of the consumer. 

The goal is to highlight the clauses in a TOS which the user would most like to read before agreeing to a TOS, without having to read the whole thing.


## Data
Our data consists of 9415 entries. it is publicly available, is based on the work of [Lippi et al (2018)](https://arxiv.org/pdf/1805.01217v2.pdf)

The data is highly unbalanced with about 10% of the data labeled as positive

## Approach
Given the fairly small sample size it is difficult to train a good deep learning model without overfitting, so I decided to approach this as a transfer learning problem.  

One possible approach would be to use a large transformer based pre-trained model, such as BERT, however I was concerned that the terms in software TOS's would not be a good fit for the text that the model was trained on.  Instead I chose to use a custom trained word2vec model, trained on a large collection of contracts.


I used data from the  [Contract Understanding Atticus Dataset (CUAD)](https://github.com/TheAtticusProject/cuad)   from the section 'Extra Data'
which consists of many GB of contracts to train a word2vec model in gensim.

I then used this word2vec model as an embedding layer in an LSTM model.

## Results and Findings
I achieved a F1 Score of 72% which is dissapointing as it is below that which was achieved by Lippi et al, however with a suitably chosen threshould I was able to achieve a recall 0f 90% (7% better than Lippi's best result) with a precision of 47% (Lower than Lippi's 72.9% Precision with their hightest recall model.)  Given the nature of this problem the increase in recall is worth a decrease in precision, i.e. it is better to read a few more standard clauses than to miss an unfair clause.  Practically speaking, this means that you can find 90% of the potentially unfair clauses while only reading 20% or the clauses in the TOS (the 10% that are potentially unfair, and another 10% that are false positives.)

I had originally intended to train the model with the embedding layer fixed (untrainable) as the word2vec model, then perhaps training this layer at the end to fine tune it.  However I discovered that I achieved much better results with the embedding layer trainable from the beginning and only using the word2vec model as and initialization.

##  Instructions for running Code
This project consists of 2 notebooks.  first run 'word2vec_model.ipynb' on a CPU with gensim=4.0.0 installed (it may work in newer versions, but the gensim api was substantially changed for 4.0.0 so older versions will not)  It is not required, but training will complete much faster if cython is installed.  You will need to update the path to where you saved the unlabeled contract data.

The second notebook 'model_and_prediction.ipynb' loads the saved word2vec model and and trains an LSTM network in Tensorflow 2.  This notebook is from Google Colab and includes statements to install required non-default packages.  Update the path to the data and the word2vec model.  

This repository contains a saved version of the LSTM model and the required tokenizer

## Future work
To extend this work I would like to try using a Transformer based model, possibly the ROBERTA model from CUAD that is fine tuned on contract data, however I would want to make sure that my data set was not among the data sets that CUAD used to train on, in order to avoid data leakage.


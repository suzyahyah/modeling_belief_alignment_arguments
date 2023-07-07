#!/usr/bin/python3
# Author: Suzanna Sia

# Standard Imports
import itertools
import os
import pickle
import json
import numpy as np
import sys
import random
from tqdm import tqdm

# Third Party
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer as tfvect

from sentence_transformers import SentenceTransformer

# Own imports
from code.data_prep import process_debates
from code.datasets import cmv_dataset, debates_dataset



# deprecated
def anneal(steps, incr, anneal_steps):

    weight =  float(1/(1+np.exp(-0.0025*(steps-anneal_steps))))
    steps += incr

    if steps > anneal_steps:
        steps = steps - anneal_steps  # restart

    if anneal_steps==0:
        weight = 1
    
    return steps, weight


#def frange_cycle_sigmoid(start=0, stop=1, n_epoch=400, n_cycle=2, ratio=1):
def frange_cycle_sigmoid(params):
#https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    n_epoch = params['n_epoch']
    n_cycle = params['n_cycle']
    stop = params['stop']
    start = params['start']
    ratio = params['ratio']

    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L    


def drop_words(ey, y_lens, word_dropout):

    unsqueeze=False
    if ey.shape[0]==1:
        ey = ey.squeeze(0)
        unsqueeze=True


    if word_dropout>0 and word_dropout<1:
        for i in range(ey.size(0)):
            drop_ = torch.rand(ey[i].size()).cuda()
            ey[i] *= (drop_ < word_dropout)

    if word_dropout==1:
        ey = torch.zeros_like(ey)

    if unsqueeze:
        ey = ey.unsqueeze(0)
    return ey

def tfidf_charts(fn):

    from matplotlib import pyplot as plt

    with open(fn, encoding='utf-8') as f:
        threads = json.load(f)


    all_pos_sims = []
    all_neg_sims = []

    for i in range(len(threads)):
        posIDs = [p2 for (p1, p2, d) in threads[i]['ID_pairs'] if d==True]
        negIDs = [p2 for (p1, p2, d) in threads[i]['ID_pairs'] if d==False]

        all_w = []
        IDs = []

        for k in threads[i]['ID_text'].keys():
            IDs.append(k)
            all_w.append((" ".join(threads[i]['ID_text'][k])))

        tfv = tfvect()
        X = tfv.fit_transform(all_w)
        X_sim = cosine_similarity(X)

        posIXs = [IDs.index(id) for id in posIDs]
        negIXs = [IDs.index(id) for id in negIDs]
        threadIX = IDs.index(threads[i]['thread_ID'])

        pos_sims = [X_sim[ix][threadIX] for ix in posIXs]
        neg_sims = [X_sim[ix][threadIX] for ix in negIXs]


        pos_sims = np.round(pos_sims, decimals=1)
        neg_sims = np.round(neg_sims, decimals=1)

        all_pos_sims.extend(pos_sims)
        all_neg_sims.extend(neg_sims)

    plt.hist(all_pos_sims)
    plt.show()

    # sentences
 

def prep_datasets(args, device, json_fn="", mode="train"):

    args = vars(args)
    args['json_fn'] = json_fn
    dataset = None
    
    print(f"mode:{mode}, fn:{json_fn}")

    if args['dataset'] == "IQ2":
        dataset = debates_dataset.DebatesDataset(args, device=device)

    if args['dataset'] == "cmv":
        dataset = cmv_dataset.CMVDataset(args, device=device)

    dataset.make_ix_dicts(args['old_emb'], args['new_emb'], args['vocab_fn'])
    dataset.proc_data()
    dataloader = dataset.get_dataloader(args['batch_size'])
    print("--prep datasets done")

    return dataset, dataloader

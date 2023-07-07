#!/usr/bin/python3
# Author: Suzanna Sia

import os
from gensim.scripts.glove2word2vec import glove2word2vec


if not os.path.exists('data/embeddings/glove_w2vec.txt'):
    _ = glove2word2vec(glove_fn, 'data/embeddings/glove_w2vec.txt')
    print("saved to :", 'data/embeddings/glove_w2vec.txt')

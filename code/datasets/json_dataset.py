#!/usr/bin/python3
# Author: Suzanna Sia

# Standard Imports
import os
import sys
import numpy as np

# Third Party
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors

from sentence_transformers import SentenceTransformer


class JSONDataset(Dataset):

    def __init__(self, args, device=None):
        super().__init__()

        self.nwords = args['nwords']
        self.encoder = args['encoder']
        self.max_seq_len = args['max_seq_len']
        self.device = device
        self.w2ix = {'<pad>':0, '<unk>':1, 'N':2, '<eos>': 3, 'DISAGREE':4, '<sos>':5}
        self.ix2w = {v:k for k, v in self.w2ix.items()}
        self.universal_embed = args['universal_embed']
        self.word_dropout = args['word_dropout']
        self.sanity_check = args['sanity_check']

        self.vocab_size = 0
        self.max_length = -1
        self.min_length = float('inf')

        print("using device:", device)
        self.encoder_model = self.get_encoder(encoder=self.encoder)


    def __len__(self):
        return len(self.data)

    def get_encoder(self, encoder="", device=""):
        if encoder == "bert":
            model = SentenceTransformer('bert-base-nli-mean-tokens')
            return model.cuda(device)

    def get_sent_v(self, sent):
        sent = sent.split()[:self.max_seq_len]
        sent = [self.w2ix[w] if w in self.w2ix else self.w2ix['<unk>'] for w in sent]
        sent = torch.LongTensor(sent).cuda() # to(self.device)

        return sent

    def __getitem__(self, ix):
        #OP, CO, delta = self.data[ix]
        return self.data[ix]


    def proc_sent(self, OP):
        OPd = {}

        sos2 = torch.LongTensor([self.w2ix['<sos>']]).cuda() 
        eos3 = torch.LongTensor([self.w2ix['<eos>']]).cuda() 

        id, OP = OP
        OP_xx = [self.get_sent_v(s) for s in OP]
        OP_xx = [x for x in OP_xx if len(x) >0]
        if len(OP_xx)==0:
            return None
        OP_ey = [torch.cat((sos2, s), dim=0) for s in OP_xx]
        OP_ye = [torch.cat((s, eos3), dim=0) for s in OP_xx]


        OPd['x_lens'] = [len(s) for s in OP_xx]
        OPd['y_lens'] = [len(s) for s in OP_ey]

        OPd['xx'] = pad_sequence(OP_xx, batch_first=True, padding_value=0)
        OPd['ye'] = pad_sequence(OP_ye, batch_first=True, padding_value=0)
        OPd['ey'] = pad_sequence(OP_ey, batch_first=True, padding_value=0)
        OPd['id'] = id

        if self.encoder!="glove":
            sents = []
            for i, op in enumerate(OP):
                sents.append(" ".join(self.replace_sw(op)))

            if self.encoder=="bert":
                OPd['bert_embed'] = torch.tensor(self.encoder_model.encode(sents)).cuda()# to(self.device)

            elif self.encoder=="infersent":
                OPd['infersent_embed'] = torch.tensor(self.encoder_model.encode(sents)).cuda() 

        return OPd


    def proc_data(self):
        #json_fn = self.json_fn[self.json_fn.rfind('/')+1:]
        data_dict = []

        for ix in range(len(self.data)):

            thread_ID, OP, all_CO_pos, all_CO_neg, all_CO_irr = self.data[ix]
            
            OP =  self.proc_sent(OP)
            if OP is None:
                continue
            
            CO_pos = [self.proc_sent(c) for c in all_CO_pos]
            CO_neg = [self.proc_sent(c) for c in all_CO_neg]
            CO_irr = [self.proc_sent(c) for c in all_CO_irr]

            CO_pos = [c for c in CO_pos if c is not None]
            CO_neg = [c for c in CO_neg if c is not None]
            CO_irr = [c for c in CO_irr if c is not None]
            data_dict.append((thread_ID, OP, CO_pos, CO_neg, CO_irr))

        self.data = data_dict

    def get_dataloader(self, batch_size=1):

        data_loader = DataLoader(dataset=self,
                                num_workers=0,
                                batch_size=batch_size,
                                shuffle=True)

        return data_loader

    def get_vocab(self, threads):
        raise NotImplementedError


    def make_ix_dicts(self, old_format_fn, new_emb_fn, vocab_fn):

        vocab_fn = vocab_fn+f".{self.nwords}"
        new_emb_fn = new_emb_fn+f".{self.nwords}"
        # construct ix from vocab

        if os.path.exists(vocab_fn):
            print("load from :", vocab_fn)
            with open(vocab_fn, 'r') as f:
                vocab = f.readlines()

            vocab = [l.strip() for l in vocab]

            i=0
            #for i, w in enumerate(lines):
            # the vocab has to correspond to the embedding later.
            # that's why we keep the index in the same order

            for w in vocab:
                self.w2ix[w] = i
                self.ix2w[i] = w
                i+=1


        else:
            print(vocab_fn, " not found.. reconstructing vocab")
            # this is the full glove txt
            # we are going to make a smaller embedding and vocab
            dirname = os.path.dirname(vocab_fn)
            if not os.path.exists(dirname):
                os.mkdir(dirname)

            embed = KeyedVectors.load_word2vec_format(old_format_fn)
            vocab_words = ['<pad>', '<unk>', 'N', '<eos>', 'DISAGREE', '<sos>']
            len_filler = len(vocab_words)

            new_embed = np.zeros((1, 300)) #zero pad
            filler = np.random.randn((len_filler-1),300) #rest of ix
            new_embed = np.vstack((new_embed, filler))

            i=0

            for w in self.vocab_words:
                if w not in embed.vocab:
                    continue

                vec = embed.word_vec(w)
                vocab_words.append(w)
                new_embed = np.vstack((new_embed, vec))
                self.w2ix[w] = i+len_filler
                self.ix2w[i+len_filler] = w
                i+=1

            vocab_txt = "\n".join(vocab_words)
            with open(vocab_fn, 'w') as f:
                f.write(vocab_txt)

            np.savetxt(new_emb_fn, new_embed, fmt="%.5f")
            print("saved new emb file:", new_emb_fn)
            print("saved new vocab file:", vocab_fn)
            sys.exit("rerun with new vocab")

        self.vocab_size = len(self.w2ix)
        print("vocab size1:", self.vocab_size)

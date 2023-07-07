#!/usr/bin/python3
# Author: Suzanna Sia

# Standard Imports
import os
import json

# Third Party
from collections import Counter
from nltk.tokenize import sent_tokenize

# Own imports
from code.data_prep import process_debates
from code.datasets.json_dataset import JSONDataset

class DebatesDataset(JSONDataset):
    
    def __init__(self, args, device="cpu"):

        super().__init__(args, device=device)
        self.sample = 10
        self.balanced = args['balanced']
        self.hyp = args['hyp']
        self.data, self.vocab_words = self.read_json_debates(args['json_fn'])

    def read_json_debates(self, fd):

        all_threads = []
        fns = os.listdir(fd)

        for fn in fns:
            fn = os.path.join(fd, fn)
            with open(fn, 'r') as f:
                thread = json.load(f)
            all_threads.append(thread)

        vocab_words = self.get_vocab(all_threads)

        data = []
        pos = 0
        neg = 0
        irr = 0

        if DEBUG_MODE:
            print("DEBUG MODE")
            all_threads = all_threads[:50]

        with open('data/IQ2_corpus/labels.txt', 'r') as f:
            labels = f.readlines()

        labels_D = {}
        for label in labels:
            label = label.strip().split('\t')
            labels_D[label[0]] = (label[1], label[2])


        pro_wins = 0
        con_wins = 0

        for i in range(len(all_threads)):
            fn = fns[i]

            thread = all_threads[i]
            thread_ID = thread['debateID']

            pros = process_debates.get_speaker(thread, 'pro')
            cons = process_debates.get_speaker(thread, 'con')
            OH = process_debates.get_speaker(thread, 'aud')

            OH = [sent_tokenize(s) for s in OH]
            OH = [s for ss in OH for s in ss]

            pros = [sent_tokenize(s) for s in pros]
            cons = [sent_tokenize(s) for s in cons]

            pros = [s for s in pros if len(s)>3]
            cons = [s for s in cons if len(s)>3]

            CO_irr = []
            CO_pos = []
            CO_neg = []


            if fn in labels_D:
                if labels_D[fn][0] == "1":
                    if labels_D[fn][1] == "pro":
                        CO_pos, CO_neg = pros, cons
                        pro_wins += 1
                    else:
                        CO_pos, CO_neg = cons, pros
                        con_wins += 1

                # if not balanced, use threads where there is no delta
                # only applies to training
                elif self.balanced == 0:
                    CO_neg = cons
                    CO_neg.extend(pros)
                else:
                    continue

            else:
                # or just treat it as no _delta.
                continue

            # get irrelevant comments (for hypothesis, and for ss_recon_loss)
            mod = process_debates.get_speaker(thread, 'mod')
            mods = [sent_tokenize(s) for s in mod]
            CO_irr = mods
 

            OH = (thread_ID, OH)
            pos += len(CO_pos)
            neg += len(CO_neg)
            irr += len(CO_irr)

            all_CO_pos = [(thread_ID, p) for p in CO_pos]
            all_CO_neg = [(thread_ID, p) for p in CO_neg]
            all_CO_irr = [(thread_ID, p) for p in CO_irr]

            data.append((thread_ID, OH, all_CO_pos, all_CO_neg, all_CO_irr))

        print(f"pro wins:{pro_wins}, con wins:{con_wins}") 
        print(f"nthreads:{len(data)} pos counts:{pos}, --neg counts:{neg} --irr counts:{irr}")
        return data, vocab_words

    def get_vocab(self, threads):

        alltext = []
        for thread in threads:
            for content in thread['content']:
                for subcontent in content['subContent']:

                    if type(subcontent) is dict:
                        alltext.append(subcontent['sentenceContent'])

                    else:
                        for sent in subcontent:
                            alltext.append(sent['sentenceContent'])


        all_words = " ".join(alltext).split()
        c_all_words = Counter(all_words).most_common(self.nwords)
        vocab_words = [c[0] for c in c_all_words]

        return vocab_words



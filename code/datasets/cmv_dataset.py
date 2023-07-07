#!/usr/bin/python3
# Author: Suzanna Sia

# Standard Imports
import json
import random

# Third Party
from collections import Counter

# Own imports
from code.datasets.json_dataset import JSONDataset 

class CMVDataset(JSONDataset):
    
    def __init__(self, args, device="cpu"):

        super().__init__(args, device=device)

        self.balanced = args['balanced']
        self.data, self.vocab_words = self.read_json_cmu(args['json_fn'])

    def _get_comment(self, thread_ID_text, thread_pairs):

        all_CO_ = []
        for pair in thread_pairs:
            co_id = pair[1]
            try:
                CO_ = thread_ID_text[co_id]
            except:
                continue

            CO_ = [s for s in CO_ if len(s.split())>5]

            if len(CO_)>0:
                all_CO_.append((co_id, CO_))

        return all_CO_    


    def read_json_cmu(self, fn):

        with open(fn, encoding="utf-8") as f:
            threads = json.load(f)
       
        vocab_words = self.get_vocab(threads)

        skip = 0
        data = []
        pos = 0
        neg = 0
        irr = 0
        
        if self.sanity_check:
            print("Sanity checking with 100 samples....")
            threads = threads[:100]
            

        for i in range(len(threads)):
            pos_pairs = list(set([(p1, p2) for (p1, p2, d) in threads[i]['ID_pairs'] if d==1]))
            neg_pairs = list(set([(p1, p2) for (p1, p2, d) in threads[i]['ID_pairs'] if d==0]))
            # can consider using less negative pairs
            n_neg_pairs = min(len(pos_pairs), len(neg_pairs))
            neg_pairs = random.sample(neg_pairs, n_neg_pairs)

            irr_pairs = [(p1, p2) for (p1, p2, d) in threads[i]['ID_pairs'] if d==2]
            nchoice = min(len(pos_pairs)+len(neg_pairs), len(irr_pairs))
            irr_pairs = random.sample(irr_pairs, nchoice)

            if self.balanced==1:
                if len(pos_pairs)==0 or len(neg_pairs)==0:
                    continue

            
            while len(irr_pairs)<(len(pos_pairs)+len(neg_pairs)):
                p1 = threads[i]['thread_ID']

                # choose a random title from the threads
                irr_text = ''
                random_thread = random.choice(threads)
                irr_text = random_thread['ID_text'][random.choice(list(random_thread['ID_text'].keys()))]
                p2 = 'irr_' + random_thread['thread_ID']

                irr_pairs.append((p1, p2))
                threads[i]['ID_text'][p2] = irr_text

            pos += len(pos_pairs)
            neg += len(neg_pairs)
            irr += len(irr_pairs)

            thread_ID = threads[i]['thread_ID']
            OH = threads[i]['ID_text'][thread_ID]
            OH = [s for s in OH if len(s.split())>5]
            OH = (thread_ID, OH)
            thread_ID_text = threads[i]['ID_text']

            all_CO_pos = self._get_comment(thread_ID_text, pos_pairs)
            all_CO_neg = self._get_comment(thread_ID_text, neg_pairs)
            all_CO_irr = self._get_comment(thread_ID_text, irr_pairs)

            data.append((thread_ID, OH, all_CO_pos, all_CO_neg, all_CO_irr))
        print(f"pos counts:{pos}, --neg counts:{neg} --irr counts:{irr}")
        return data, vocab_words


    def get_vocab(self, threads):
        # get all words in vocab that appear at least 3 times
        all_words = []
        for i in range(len(threads)):
            all_words.append(threads[i]['title'])
            texts = list(threads[i]['ID_text'].values())

            for t in texts:
                all_words.append("\n".join(t))

        all_words = "\n".join(all_words).split()
        c_all_words = Counter(all_words).most_common(self.nwords)
        vocab_words = [c[0] for c in c_all_words]
        
        return vocab_words

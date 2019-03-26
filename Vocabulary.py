
from nltk import word_tokenize
from collections import defaultdict
import string
class Vocabulary:
    def __init__(self, special_tokens=None):
        self.unk_count = 0
        self.total_token_count = 0
        self.w2idx = {}
        self.idx2w = {}
        self.w2cnt = defaultdict(int)
        self.special_tokens = special_tokens
        if self.special_tokens is not None:
            self.add_tokens(special_tokens)

    def unk_ratio(self):
        return f'{self.unk_count}/{self.total_token_count}({(self.unk_count/self.total_token_count)*100:.3f} %)'

    def get_sentence(self,sentence):
        id_list = []
        for word in sentence:
            try:
                self.total_token_count += 1 
                id_list.append(self[word])
            except KeyError:
                self.unk_count+=1
                id_list.append(self['<UNK>'])
        return id_list

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)
            self.w2cnt[token] += 1

    def add_token(self, token):
        if token not in self.w2idx:
            cur_len = len(self)
            self.w2idx[token] = cur_len
            self.idx2w[cur_len] = token

    def prune(self, min_cnt=2):
        to_remove = set([token for token in self.w2idx if self.w2cnt[token] < min_cnt])
        if self.special_tokens is not None:
            to_remove = to_remove.difference(set(self.special_tokens))
        
        for token in to_remove:
            self.w2cnt.pop(token)
            
        self.w2idx = {token: idx for idx, token in enumerate(self.w2cnt.keys())}
        self.idx2w = {idx: token for token, idx in self.w2idx.items()}
    
    def __contains__(self, item):
        return item in self.w2idx
        
    def convert_to_list(self):
        l = []
        for i in range(len(self.idx2w)):
            l.append(self.idx2w[i])
        return l

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.w2idx[item]
        elif isinstance(item , int):
            return self.idx2w[item]
        else:
            raise TypeError("Supported indices are int and str")
    
    def __len__(self):
        return(len(self.w2idx))

def preprocess(data):
    """
    Args:
        data (str):
    Returns: a list of tokens

    """
    tokens = [t for t in word_tokenize(data) if t not in string.punctuation]
    return tokens

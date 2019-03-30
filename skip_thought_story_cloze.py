#!/usr/bin/env python
# coding: utf-8

import torch

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import sys
from skipthoughts import UniSkip,BiSkip
import pandas as pd
from Vocabulary import Vocabulary, preprocess
import torch.optim as optim
from model import Net

dir_st = 'data/skip-thoughts'
sys.path.append('skip-thoughts.torch/pytorch')

def make_vocab(tokens):
    voc = Vocabulary(['<UNK>'])
    voc.add_tokens(tokens)
    print('vocab len is {}'.format(len(voc.w2idx)))
    return voc



def load_data(file='story_cloze_data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv'):
    '''TODO remove 10% of data for hyper param tuning'''
    df= pd.read_csv(file)
    df = df.drop('InputStoryid',axis=1)
    targets = df['AnswerRightEnding']
    df = df.drop('AnswerRightEnding',axis=1)
    df = df.drop('InputSentence1',axis=1)
    df = df.drop('InputSentence2',axis=1)
    df = df.drop('InputSentence3',axis=1)
    
    voc_str= ''
    for index, row in df.iterrows():
        voc_str+=' '.join(list(row)) + ' '
        
    df['AnswerRightEnding'] = targets
    return df, make_vocab(preprocess(voc_str))


class LastSentenceDataset(Dataset):
    '''currently implements no context model. will add in last sentence later'''
    def __init__(self,file='story_cloze_data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv',vocab=None):

        super().__init__()
        
        df, created_vocab = load_data(file)
        
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = created_vocab
        self.df = df
      
        
        self.dir_st = 'data/skip-thoughts'
        self.biskip = BiSkip(self.dir_st, self.vocab.convert_to_list())
        
        self.uniskip = UniSkip(self.dir_st, self.vocab.convert_to_list())
        
       
        self.data = self.make_data()
        
        
    def __getitem__(self, idx):
        """
        Args:
            idx
        Returns: skip thought embedding of ending and 0/1 if it is the right ending 

        """
        return self.data[idx]

    def __len__(self):
        """
        Returns len of the dataset
        """
        return len(self.data)
       
    def make_data(self):
        data = []
        total = self.df.index[:100]
        print('skip thought encoding dataset')
        for i in total:
            #print(row['RandomFifthSentenceQuiz1'],row['RandomFifthSentenceQuiz2'])
           
            self.progress(i,len(total))
            endings =  self.gen_embbeding(self.df.at[i,'RandomFifthSentenceQuiz1'], self.df.at[i,'RandomFifthSentenceQuiz2'])
            if self.df.at[i,'AnswerRightEnding'] == 1:
                data.append((endings[0],torch.tensor([1])))
                data.append((endings[1],torch.tensor([0])))
            else:
                data.append((endings[0],torch.tensor([0])))
                data.append((endings[1],torch.tensor([1])))
        print(f'\n{self.vocab.unk_ratio()} ids generated were replaced with <UNK>')
        return data
    

    def zero_pad(self,l,n):
        l = (l + n * [0])[:n]
        return l
    
    def pad_input(self,a,b):
        ed = sorted([a,b],key=len)
        longer = ed[1]
        shorter = ed[0]
        padded = self.zero_pad( shorter,len(longer))
        if shorter == a:
            return padded,b
        else: return a,padded
        
    def gen_embbeding(self,sent1,sent2):
        sent1 = preprocess(sent1)
        sent2 = preprocess(sent2)
        #remove random n token that is in one sentence
        if 'n' in sent2:
            sent2.remove('n')
        encoded_end1 = self.vocab.get_sentence(sent1)
        encoded_end2 = self.vocab.get_sentence(sent2)
        a,b = self.pad_input(encoded_end1,encoded_end2)
        
        batch = Variable(torch.LongTensor([a,b])) 
        top_half = self.uniskip(batch)
        bottom_half = self.biskip(batch)
        combine_skip = torch.cat([top_half,bottom_half],dim=1) 
        return combine_skip
    
    def progress(self,count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

if __name__ == "__main__":
    #the Dataset uses a lot of memory be carefull
    val_data_set= LastSentenceDataset()
    print('train 1',val_data_set[0])

    net = Net()
    #exit beacuse the training loop is broken
    sys.exit()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    running_loss = 0.0
    for i, data in enumerate(val_data_set, 0):
       # get the inputs
        inputs, labels = data
        print(inputs,labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)
           
        print('out',outputs,labels)
        #TODO Fix this 
        # posible solution https://github.com/pytorch/pytorch/issues/5554
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        print('Finished Training')



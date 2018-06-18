import torch
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import collections

class Vocab():
    def __init__(self,data_dir):
        self.data_dir =data_dir
        self.train_len = 0
        self.valid_len = 0
        self.test_len = 0

    def preprocess(self):
        self.train_name = self.data_dir +"/train.txt"
        self.valid_name = self.data_dir +"/valid.txt"
        self.test_name = self.data_dir +"/test.txt"

        if not (os.path.exists(self.train_name) and os.path.exists(self.valid_name)):
            f1 = open(self.train_name, "w")
            f2 = open(self.valid_name, "w")
            f3 = open(self.test_name, "w")
            file_name =self.data_dir +"/input.txt"
            count =0
            with open(file_name,"r") as f:
                for line in f:
                    count+=1
                    if count==9:
                        f2.write(line)
                        self.valid_len += 1
                    elif count ==10:
                        f3.write(line)
                        self.test_len+=1
                        count =0
                    else:
                        f1.write(line)
                        self.train_len += 1
            f1.close()
            f2.close()
            f3.close()
        else:
            self.train_len = sum(1 for line in open(self.train_name))
        return

    def create_vocab(self):
        file_name =self.data_dir +"/input.txt"
        count = 0
        with open(file_name,"r") as f:
            data =f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars,range(len(self.chars))))
        #print("vocab",self.vocab)
        return

    def char_index(self,chars):
        return Variable(torch.LongTensor([self.vocab[c] for c in chars]).view(1,-1))


class TextLoader():
    def __init__(self,batch_size=256,seq_length=200):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.T = Vocab("data")
        self.T.preprocess()
        self.T.create_vocab()
        self.current =0
        self.n_batch =100
        self.x_batches =[]
        self.y_batches = None

    def loaddata(self, name):

        f1 =open(self.T.train_name,"r")
        data = f1.read()
        xdata = np.array(list(map(self.T.vocab.get, data)))

        self.n_batch = len(xdata)//(self.batch_size*self.seq_length)
        print(" here n_batch", self.n_batch, self.batch_size,self.seq_length)
        total_len = self.n_batch * self.batch_size*self.seq_length
        ydata = xdata[1:total_len+1]
        xdata = xdata[:total_len]
        if len(ydata) < len(xdata):
            ydata[total_len] = xdata[0]

        self.x_batches = xdata.reshape(self.n_batch,self.batch_size,self.seq_length)
        self.y_batches = ydata.reshape(self.n_batch,self.batch_size,self.seq_length)

        f1.close()

    def loadvalid(self):
        f1 =open(self.T.valid_name,"r")
        f1.close()

    def get_next(self):
        #print(" get next",self.n_batch)
        if self.current == self.n_batch-1:
            self.current = 0
            self.current +=1
        x = self.x_batches[self.current-1]
        return torch.LongTensor(x),torch.LongTensor(self.y_batches[self.current-1])

'''
if __name__ == '__main__':
    t = TextLoader()
    t.loaddata('train')
    print(t.get_next())
    for i in range(t.n_batch):
        print('***********')
        print(t.x_batches[i],t.y_batches[i])
'''

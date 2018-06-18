import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import data
import argparse


class CharRNN(nn.Module):
    def __init__(self,embedding_size, hidden_dim,vocab_size,num_layers=1):
        super(self.__class__, self).__init__()
        self.embed_dim = embedding_size
        self.hidden_dim = hidden_dim
        self.encode= nn.Embedding(vocab_size, embed_dim)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.decode = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        inp = self.encode(input)
        output, hidden = self.lstm(inp, hidden)
        output = self.decode(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)))

class CharModel():
    def __init__(self,embed_size,hidden_dim,num_layers):
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers



    def train_epoch(self):
        hidden = self.model.init_hidden(self.batch_size)
        self.model.zero_grad()
        loss = 0
        x,y = self.data.get_next()

        out, hidden = self.model(x,hidden)
        loss =self.criterion(out.view(-1,self.vocab_size),y.view(-1))
        loss.backward()
        self.optimizer.step()

        return loss.data[0]



    def train(self,batch_size,epochs,seq_length=200):
        self.batch_size = batch_size
        self.data = data.TextLoader(batch_size,seq_length)
        self.data.loaddata('train')
        self.vocab_size = self.data.T.vocab_size
        self.args = (self.vocab_size,self.embed_size,self.hidden_dim)
        self.model = CharRNN(self.embed_size,self.hidden_dim,self.vocab_size,self.num_layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        hidden = self.model.init_hidden(batch_size)
        n_epochs =epochs
        losses =[]
        avg_loss = 0

        for epoch in range(n_epochs):
            loss = self.train_epoch()
            avg_loss += loss

            if epoch %10 == 0:
                print('epoch_no', epoch,'loss',loss)
                print(self.run('\n', 150, 0.5), '\n')
                losses.append(avg_loss/10.0)
                avg_loss = 0
        self.save()
        return losses

    def save(self):
        state = {
        'args':self.args,
        'state_dict': self.model.state_dict()
        }

        torch.save(state,'model/model.pth.tar')
        print("Saved")

    def load(self):
        state = torch.load('model/model.pth.tar')
        self.args = state['args']
        self.model = CharRNN(*self.args)
        self.model.load_state_dict(['state_dict'])


    def run(self, init_str='A', length=200, temp=0.4):
        hidden = self.model.init_hidden(1)
        batch_size=1
        pred = init_str
        self.Vocab = data.TextLoader(batch_size,length)

        if len(init_str) > 1:
            input = self.Vocab.T.char_index(init_str[:-1])
            _, hidden = model(input, hidden)

        #print("gygsgab",self.Vocab.T.vocab['\n'])
        input =self.Vocab.T.char_index(init_str[-1])
        #print('input',input.data)

        inv_dict = dict((v,k) for k, v in self.Vocab.T.vocab.items())

        for i in range(length):
            output, hidden = self.model(input, hidden)
            output_dist = F.softmax(output.view(-1)/temp, dim=0).data
            idx = torch.multinomial(output_dist, 1)[0]
            pred_char = inv_dict[int(idx)]
            pred += pred_char
            input = self.Vocab.T.char_index(pred_char)
        return pred




if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pretrained', type =bool, default = False)
    parser.add_argument('-l','--text_length', type =int, default = 500)
    args = parser.parse_args()

    embed_dim = 128
    hidden_dim = 128

    model = CharModel(embed_dim, hidden_dim, 1)

    batch_size = 64
    epochs = 2000
    if not args.pretrained:
        losses = model.train(batch_size, epochs)
    # can plot losses
    length = 500
    print(model.run('\n', length, 0.3))

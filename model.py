import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
old sources: https://medium.com/@stepanulyanin/captioning-images-with-pytorch-bc592e5fd1a3
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

new source:
https://github.com/muhyun/image_captioning/blob/master/model.py
'''

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_size #set the hidden size
         
        self.embed = nn.Embedding(vocab_size, embed_size) #the embedded layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) #lstm cell
        self.linear = nn.Linear(hidden_size, vocab_size) #linear layer after lstm
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size)) #hidden layer, start off by instantiating to 0
        
    
    def forward(self, features, captions):
        caption_embedding = self.embed(captions[:,:-1]) #get the embeddings
        embedding = torch.cat((features.unsqueeze(1), caption_embedding), 1) #goes through all of the features
        lstm_out, self.hidden = self.lstm(embedding) #passes the embeddigns through the lstm and gets the output and next hidden state
        output = self.linear(lstm_out) #passes throug the linear layer
        
        return output #return the output
    
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        hidden = None #for the first loop, the hidden state is set to None
        for i in range(max_len): #we want to go for a max of 20 times
            output, hidden = self.lstm(inputs, hidden)
            output = self.linear(output.squeeze(1))
            target_index = output.max(1)[1] #this gets the max value from the output, which is the next word
            sentence.append(target_index.item()) #appens the integer representation for the word
            inputs = self.embed(target_index).unsqueeze(1) #the output is also the next input
        return sentence

  
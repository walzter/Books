#!/usr/bin/env/python3

# here we will define the sentiment classifier 

import json 

import torch
from torch import nn
from transformers import BertModel

CONFIG_PATH = 'SentimentAnalyzer/config.json'

# reading the config file 
with open(CONFIG_PATH) as f:
    config = json.load(f)
    

# we define the class for the sentiment classifer

class BertClassifies(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifies, self).__init__()
        # load the model
        self.model = BertModel.from_pretrained(config['BERT_MODEL'],state_dict=torch.load(config['PRE_TRAINED_MODEL'],map_location=torch.device('cpu')))
        # apply some dropout 
        self.drop  = nn.Dropout(p=0.3)
        # classifier 
        self.out   = nn.Linear(self.model.config.hidden_size, n_classes) 
    
    def forward(self, input_ids, attention_mask):
        # here we need to separate the pool 
        out = self.model(input_ids = input_ids,
                        attention_mask = attention_mask)
        # pool out 
        pool_out = out['pooler_output'] # the last value is the pooled_out
        # passing it through dropout 
        out = self.drop(pool_out)
        return self.out(out)
        
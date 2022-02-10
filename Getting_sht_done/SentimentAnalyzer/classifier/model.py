#!/usr/bin/env/python3

# we load the config
import json 

import torch 
import torch.nn.functional as F 
from transformers import BertTokenizer

# load the classifier 
from .feel_classifier import BertClassifies

CONFIG_PATH = 'SentimentAnalyzer/config.json'
# load the config 
with open(CONFIG_PATH) as f: 
    config = json.load(f)


# we now create the class Model 

class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(config['BERT_MODEL'])
        
        # the classifier 
        classifier = BertClassifies(len(config['CLASS_NAMES']))
        # load it 
        #m = torch.load(config['PRE_TRAINED_MODEL'], map_location=torch.device('cpu'))
        #classifier.load_state_dict(m)
        # eval mode
        classifier.eval()
        
        self.classifier = classifier

    # predictions 
    def predict(self, text):
        # tokenizing 
        encoder = self.tokenizer.encode_plus(
            text,
            max_length = config['MAX_SEQ_LENGTH'],
            add_special_tokens = True, 
            return_token_type_ids = False, 
            pad_to_max_length = True, 
            return_attention_mask = True, 
            return_tensors = 'pt',
        )
        input_ids      = encoder['input_ids']
        attention_mask = encoder['attention_mask']
        
        # predicting 
        with torch.no_grad():
            probas = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        
        # confidence and predicted class 
        confidence, pred_class = torch.max(probas, dim=1)
        # probas 
        probas = probas.flatten().numpy().tolist()
        # combine class names with probabilities 
        class_probas = dict(zip(config['CLASS_NAMES'], probas))
        # make the dictionary 
        response_dict = (
                            config['CLASS_NAMES'][pred_class],
                            confidence, 
                            class_probas
                        )
        return response_dict
    
# instantiate the model 
model = Model()

# return the model 
def get_model():
    return model

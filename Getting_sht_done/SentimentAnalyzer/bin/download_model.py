#!/usr/bin/env/python3

import gdown
# we need to download the model just in case 

MODEL_PATH = "https://drive.google.com/uc?id=1V8itWtowCYnb2Bc9KlK9SxGff9WwmogA"

PATH_TO_DOWNLOAD = '/Users/Eric/Documents/Python/Reading/ML/PyTorch/Notes/Getting_sht_done/SentimentAnalyzer/assets/model_state_dict.bin'

# downloading the model 
gdown.download(MODEL_PATH,
               PATH_TO_DOWNLOAD)

# verbos
print(f"Downloaded the model to {PATH_TO_DOWNLOAD}")

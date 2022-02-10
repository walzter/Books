#!/usr/bin/env/python3

# imports 
from typing import Dict

# fapi
from fastapi import Depends, FastAPI
# pydantic
from pydantic import BaseModel

from .classifier.model import Model, get_model

# instantiate the app 
app = FastAPI()

# create the different classes 

# request --> Giving the text 
class FeelingRequest(BaseModel):
    text: str
    
# receiving 
class FeelingResponse(BaseModel):
    
    # we will output a dictionary
    # with probabilities 
    probabilities: Dict[str, float]
    # the sentiment 
    sentiment: str
    # confidence 
    confidence: float

# now we will post to the app 

@app.post("/predict",response_model=FeelingResponse)
# define a prediction 
def predict(request: FeelingRequest, model: Model=Depends(get_model)):
    sentiment, confidence, proba = model.predict(request.text)
    return FeelingResponse(
        sentiment = sentiment, confidence = confidence, probabilities = proba
    )
# It works! 
# curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"text": "This is a very very BAD BAD day"}'


# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, './'))
from masked_roberta import RobertaForSequenceClassification, RobertaModel
import masked_roberta_fine


def MaskedRoberta():
    '''
    This partitions the model into two parts, i.e., 
    embedding layers, encoder layers. Calculate the threshold for each part.
    '''
    pt_model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=4)
    model = RobertaForSequenceClassification(pt_model.config)
    # print(model.state_dict().keys())
    # print(pt_model.state_dict().keys())
    model.load_state_dict(pt_model.state_dict())
    return model 


def MaskedRoberta_fine():
    '''
    A key difference with MaskedRoberta() is that 
    the thresholds vary among encoder layers.
    '''
    pt_model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=4)
    model = masked_roberta_fine.RobertaForSequenceClassification(
        pt_model.config)
    # print(model.state_dict().keys())
    # print(pt_model.state_dict().keys())
    model.load_state_dict(pt_model.state_dict())
    return model 


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return tokenizer

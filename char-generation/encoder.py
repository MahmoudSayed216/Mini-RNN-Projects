from Mapper import Mapper
import torch
from configs import * 

def encode(char, dictionary = None):
    if dictionary is None:
        dictionary = Mapper.char2idx

    idx = dictionary[char]
    encoding = torch.zeros(EMBEDDING_SIZE)
    encoding[idx] = 1.0
    return encoding

def encode_example(sentence, dictionary = None):
    encodings = []
    for char in sentence:
        encodings.append(encode(char, dictionary))
    
    return encodings
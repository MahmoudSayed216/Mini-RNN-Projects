import torch
from torch.utils.data import Dataset
from Mapper import Mapper
import encoder
from configs import *


class RNNDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()

        file = open(data_path)
        self.sentences = file.readlines()
        file.close()

        self.sentences = [s.strip('\n').lower() for s in self.sentences]
        full_data = "".join(self.sentences).lower()
        unique_chars = set((full_data))
        Mapper.map(unique_chars)
        self.sentences = self.pad_strings(self.sentences)

    def __len__(self):
        return len(self.sentences)
        
    
    def __getitem__(self, index):
        sentence = self.sentences[index]

        encoded = encoder.encode_example(sentence)
        X = torch.stack(encoded[:-1])
        Y = torch.stack(encoded[1:]).argmax(dim=-1)

        return X, Y



    def pad_strings(self, strings):
        max_len = len(max(strings, key=lambda x : len(x)))
        
        padding_sizes = [max_len - len(seq) + 1 for seq in strings]
        for i, size in enumerate(padding_sizes):
            strings[i] = strings[i] + "".join(EOS*size)

        return strings
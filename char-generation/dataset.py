import torch
from torch.utils.data import Dataset
from Mapper import Mapper


class RNNDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()

        file = open(data_path)
        self.sentences = file.readlines()
        file.close()

        self.sentences = [s.strip('\n').lower() for s in self.sentences]
        self.sentences = self.pad_strings(self.sentences)
        
        full_data = "".join(self.sentences)
        unique_chars = set((full_data).lower())
        self.embedding_size = len(unique_chars)
        print("Embedding size: ",self.embedding_size)
        Mapper.map(unique_chars)
        self.char2idx = Mapper.char2idx
        self.padding_idx = self.char2idx['0']
        print(self.char2idx)

    def __len__(self):
        return len(self.sentences)
        
    
    def __getitem__(self, index):
        sentence = self.sentences[index]

        encoded = self.encode_example(sentence)
        X = torch.stack(encoded[:-1])
        Y = torch.stack(encoded[1:]).argmax(dim=-1)

        return X, Y

    def encode(self, char):
        idx = self.char2idx[char]
        encoding = torch.zeros(self.embedding_size)
        encoding[idx] = 1.0
        return encoding
    
    def encode_example(self, sentence):
        encodings = []
        for char in sentence:
            encodings.append(self.encode(char))
        
        Y = torch.stack(encodings[1:])
        return encodings

    def pad_strings(self, strings):
        max_len = max(strings, key=lambda x : len(x))
        
        padding_sizes = [len(max_len) - len(seq)+1 for seq in strings]
        for i, size in enumerate(padding_sizes):
            strings[i] = strings[i] + "".join('0'*size)

        return strings
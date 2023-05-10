
from torchcrf import CRF
import torch.nn as nn
class BiLSTM_CRF(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True
        )
        self.cnn = nn.Sequential(
            nn.Linear(hidden_size * 2, num_classes)
        )
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, x, mask):
        pred, _ = self.lstm(x)
        pred = self.cnn(pred)
        pred = self.crf.decode(pred, mask=mask)
        return pred
    
    def loss(self, x, y, mask):
        pred, _ = self.lstm(x)
        pred = self.cnn(pred)
        loss = (- self.crf(pred, y, mask=mask, reduction='mean'))
        return loss

    def predict(self, x):
        return self.forward(x)
    
    def predict_proba(self, x):
        return self.forward(x)
    

# Standard library imports
from typing import List, Tuple, Dict
from math import ceil
import re

# Third-party library imports
import gensim.downloader as api
import numpy as np
import torch
from torch.nn.functional import one_hot
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import nltk
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Project-specific imports
PTB_MAP = {'!': '.', '#': '.', '$': '.', "''": '.', '(': '.', ')': '.', ',': '.', '.': '.', ':': '.', '?': '.', 'CC': 'CONJ', 'CD': 'NUM', 'DT': 'DET', 'EX': 'DET', 'FW': 'X', 'IN': 'ADP', 'JJ': 'ADJ', 'JJR': 'ADJ', 'JJRJR': 'ADJ', 'JJS': 'ADJ', 'LS': 'X', 'MD': 'MD', 'NN': 'NN', 'NNP': 'NNP', 'NNPS': 'NNPS', 'NNS': 'NNS', 'NP': 'NP', 'PDT': 'DET', 'POS': 'POS', 'PRP': 'PRON', 'PRP$': 'PRON', 'PRT': 'PRT', 'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'RN': 'X', 'RP': 'RP', 'SYM': 'X', 'TO': 'TO', 'UH': 'X', 'VB': 'VB', 'VBD': 'VBD', 'VBG': 'VBG', 'VBN': 'VBN', 'VBP': 'VBP', 'VBZ': 'VBZ', 'VP': 'VP', 'WDT': 'DET', 'WH': 'X', 'WP': 'PRON', 'WP$': 'PRON', 'WRB': 'ADV', '``': '.'}

# Constants
PTB_DICT = {item: i+1 for i, item in enumerate(sorted(set(PTB_MAP.values())))}
PTB_DICT[None] = 0
NUM_POS_TAGS = len(set(PTB_MAP.values()))
english_stopwords = set(stopwords.words('english'))

class EventDetectionEvaluation(torch.utils.data.Dataset):
    def __init__(self, tokens: List[List[str]], wv: KeyedVectors, window_size: int = 30, window_shift: int = 10):
        # Assigning the variables
        self.tokens = tokens
        self.wv = wv

        # Dictionary for words
        words = set([word for line in tokens for word in line])
        self.word_dict, self.inv_word_dict = self.create_dicts(words)

        # Creating the windows
        self.data = self.create_windows(tokens, window_size, window_shift)
        
        # Tokens and POS tags
        self.token_windows = []
        self.pos_windows = []

        for sample in self.data:
            self.token_windows.append([[self.word_dict[word] for word in window['tokens']] for window in sample])
            self.pos_windows.append([[PTB_DICT[pos] for pos in window['pos_tags']] for window in sample])

        # List of Tensor Features
        self.features = []
        for token_window, pos_window in zip(self.token_windows, self.pos_windows):
            self.features.append(torch.tensor(
                [[self.get_features(word_idx, tag_idx) 
                        for word_idx, tag_idx in zip(token, pos_tag)] 
                        for token, pos_tag in zip(token_window, pos_window)]
            ))

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx])
    
    
    def create_windows(self, tokens: List[str], window_size: int, window_shift: int) -> List[List[int]]:
        """Processes the text into windows"""
        data = []
        for line in tokens:
            # List of POS tags
            pos = [PTB_MAP[tag[1]] for tag in pos_tag(line)]
            d = []
            for i in range(max(0, ceil((len(line) - window_size) / window_shift)) + 1):
                tokens = line[i*window_shift:i*window_shift+window_size]
                pos_tags = pos[i*window_shift:i*window_shift+window_size]

                if len(tokens) < window_size:
                    tokens += [None] * (window_size - len(tokens))
                    pos_tags += [None] * (window_size - len(pos_tags))

                d.append({'tokens': tokens, 'pos_tags': pos_tags})

            data.append(d)

        return data
    
    def create_dicts(self, items: set) -> Tuple[Dict, Dict]:
        """Creates a dictionary and its inverse from a set"""
        d = {item: index+1 for index, item in enumerate(items)}
        d[None] = 0
        inv_d = {v: k for k, v in d.items()}
        return d, inv_d

    def get_features(self, word_idx: int, tag_idx: int) -> List[float]:
        """returns the features for a word"""
        p = self.wv.vector_size + len(self.additonal_features('hello')) + NUM_POS_TAGS
        # Index 0 is the padding
        if word_idx == 0:
            return [0] * p
        else:
            word = self.inv_word_dict[word_idx]
            embeds = self.embed_word(word, self.wv)
            tag_dummy = [0] * NUM_POS_TAGS
            tag_dummy[tag_idx-1] = 1
            return list(embeds) + self.additonal_features(word) + tag_dummy
        
    def additonal_features(self, word: str) -> List[int]:
        """returns additional features for a word"""
        return [len(word), int(word[0].isupper()), int(re.search('[a-zA-Z]', word) is None), int(word in english_stopwords)]
        
    def embed_singular(self, word, wv):
        temp = word.lower()
        if temp in wv:
            return wv[temp], False
        elif any(char.isdigit() for char in temp) and (not any(char.isalpha() for char in temp) or temp[-2:] in ['th', 'st', 'nd', 'rd']):
            return wv['<number>'], False
        else:
            return wv['<unknown>'], True
        
    def embed_word(self, token, wv):
        embedding, is_unk = self.embed_singular(token, wv)

        if is_unk:
            # Initialize embedding to zero
            embedding = np.zeros(wv.vector_size, dtype=np.float32)
            # Split token into subwords
            subwords = re.split(r'-|_|/', token)
            for subword in subwords:
                embedding += self.embed_singular(subword, wv)[0]
            
            embedding /= len(subwords)

        return embedding

    def get_pos_features(tags: List[int]) -> torch.Tensor:
        return one_hot(torch.tensor(tags), num_classes=len(set(PTB_MAP.values())))
import numpy as np
from typing import List

from model import Model


def build_model(device: str) -> Model:
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)


class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"), 
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]

import torch
import torch.nn as nn
import pickle


import sys
sys.path.append('model')
# utils is in the model
from utils import BiLSTM_CRF
from utils import EventDetectionEvaluation


class StudentModel(Model):
    def __init__(self, device: str):
        with open('model/wv.pkl', 'rb') as f:
            self.wv = pickle.load(f)
        print('Embeds loaded')
        # Inverse Dictionary
        self.inv_superlabel_dict = {
            1: 'POSSESSION', 2: 'O', 3: 'ACTION', 4: 'SCENARIO', 5: 'SENTIMENT', 6: 'CHANGE', 0: None
        }

        # Model Setup
        n_features = 130
        n_classes = 6
        n_hidden = 32
        n_layers = 2

        self.best_model = BiLSTM_CRF(n_features, n_hidden, n_layers, n_classes)
        self.best_model.load_state_dict(torch.load('model/best_model_crf_32_2.pt'))
        self.best_model.to(device)
        self.best_model.eval()
        print('Model loaded')


    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!

        # Dataset Setup
        dataset = EventDetectionEvaluation(tokens, self.wv)
        print('Dataset created')

        # Predictions
        predictions = []
        for features in dataset:
            mask = features.sum(dim=2) != 0
            predictions.append(self.get_superlabels_fix(self.predict_labels(features, mask, self.best_model)))
        print('Predictions done')
        return predictions

    def predict_labels(self, features, mask, model) -> List[int]:
        predictions = model(features, mask)
        y_pred = []
        for i, pred in enumerate(predictions):
            if i < len(predictions) - 1:
                # window shift
                y_pred += pred[:10]
            else:
                y_pred += pred

        return y_pred
    
    def get_superlabels_fix(self, tensor: List[int]) -> List[str]:
        """Converts BIO labels to superlabels"""

        temp = [self.inv_superlabel_dict[i+1] for i in tensor]
        for i in range(len(temp)):
            if temp[i] != 'O':
                if i > 0 and temp[i-1][2:] == temp[i]:
                    temp[i] = 'I-' + temp[i]
                else:
                    temp[i] = 'B-' + temp[i]
        return temp
import csv
import pickle
import os
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset

output = '../data/new.tsv'
training_s = 0.7
validation_s = 0.1
testing_s = 0.2

def train_valid_test(dataset):
    split_index = math.floor(len(dataset),training_s)
    train = dataset[split_index:]
    test = dataset[:split_index]
    return train, test


"""Convert CSV file to TSV file"""
def makeTSV(path: str)->None:
    with open(path, 'r') as csvin, open(output, 'w') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout, delimiter='\t')

        for row in csvin:
            tsvout.writerow(row)

"""Extract claims, evidence, and labels as discrete lists with index matching"""
def prepareData(path: str):
    #df = pd.read_csv(path, sep='\t', header=None)
    evidences, claims, labels = [], [], []

    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line_i, line in enumerate(lines):
            content = line.split("\t")
            text = content[0]
            evidence = content[1]
            label = content[2]
            claims.append(text)
            evidences.append(evidence)
            labels.append(label)
    
    return claims, evidences, labels

# Reference: https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
class MyDataset(Dataset):
    def __init__(self, data, targets=None, transform=None, transform_target=None):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float() if targets is not None else None
        self.transform = transform
        self.transform_target = transform_target
        
    def __getitem__(self, index):
        x = self.data[index]
        y = np.zeros(1, dtype=float)# self.targets[index]

        if self.targets is not None:
            y = self.targets[index]
        else:
            None
        if self.transform:
            x = self.transform(x)
        if self.transform_target:
            y = self.transform_target(y)
        
        return x, y
    
    
    def __len__(self):
        return len(self.data)

if __name__=="__main__":
    makeTSV('../data/dataset.csv')
    prepareData(output)
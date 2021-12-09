import csv
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

input = '../data/dataset.csv'
output = '../data/new.tsv'
training_s = 0.7
validation_s = 0.1
testing_s = 0.2

def makeDF(path: str):
    dataset = pd.read_csv(path, sep='\t', header=None,dtype='str')
    return dataset

"""Convert CSV file to TSV file"""
def makeTSV(path: str)->None:
    with open(path, 'r') as csvin, open(output, 'w') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout, delimiter='\t')

        for row in csvin:
            tsvout.writerow(row)

"""Split data into training, validation, and test set"""
def train_valid_test(dataset: pd.DataFrame):
    temp = len(dataset)
    split_index = math.floor(len(dataset)*training_s)
    test = dataset[int(temp*0.8):]
    valid = dataset[split_index:split_index+int(validation_s*temp)]
    train = dataset[:split_index+int(0.1*temp)]
    print(f'Dataset shapes (dataset, train, val, test): {dataset.shape}, {train.shape},   {valid.shape}, {test.shape}')
    return train, valid, test

"""Extract claims, evidence, and labels as discrete lists with index matching"""
def splitData(df: pd.DataFrame):
    return df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]

class MyDataset(Dataset):
    def __init__(self, claim, data, labels):
        self.claim = claim
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.claim[index]
        y = self.data[index]
        z = self.labels[index]
        w = {"Claim": x, "Evidence": y, "Label": z}
        return w
    
class singleData(Dataset):
    def __init__(self, data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x = self.data[index]
        return x

def makeGenerator(v: pd.DataFrame):
    a, b, c = splitData(v)
    generator = torch.utils.data.DataLoader(MyDataset(a,b,c), batch_size=1)
    
    return generator

def makeSubscriptable(v: pd.DataFrame):
    a, b, c = splitData(v)
    subs1 = torch.utils.data.DataLoader(singleData(a), batch_size=1)
    subs2 = torch.utils.data.DataLoader(singleData(b), batch_size=1)
    subs3 = torch.utils.data.DataLoader(singleData(c), batch_size=1)
    return subs1, subs2, subs3

def __ExportDataset__():
    makeTSV(input)
    t, v, t_ = train_valid_test(makeDF(output))
    _, _, o = splitData(t)
    w = torch.tensor(np.array([0.5 for v in range(len(o))]))
    print("New dataset successfully loaded! ... \n")
    optim = makeSubscriptable(t)
    return makeGenerator(t), makeGenerator(v), makeGenerator(t_), w, optim

if __name__=='__main__':
    __ExportDataset__()
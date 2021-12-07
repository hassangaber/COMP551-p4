import csv
import pickle
import os
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset

output = '../data/new.tsv'
accepted_labels = [0, 1]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

"""Convert CSV file to TSV file"""
def makeTSV(path: str)->None:
    with open(path, 'r') as csvin, open(output, 'w') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout, delimiter='\t')

        for row in csvin:
            tsvout.writerow(row)

"""
Data here is seperated as:
Claim   [10 instances of evidence]  Label
Each column is seperated by a `\t`
"""
makeTSV('../data/dataset.csv')

# -*- coding: utf-8 -*-
"""scrape.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PUrEBBmplLvJY1yOgspsptEf8BfeE8Fx
"""

import pandas as pd
from googleapiclient.discovery import build

API_KEY = "AIzaSyCyQIwmrJjCvGGEbQfREC8jLMMI6scafcg"
CSE_ID = "542c95b633b0e06e0"

claimsDf = pd.read_csv('train.csv').drop(["Text_Tag"], axis=1)

# Convert Mostly-True to True
claimsDf.Labels = claimsDf.Labels.replace(3, 5)

# Convert Barely-True to False
claimsDf.Labels = claimsDf.Labels.replace(0, 1)

claimsDf = claimsDf.loc[claimsDf['Labels'].isin([1, 5])].reset_index().drop(["index"], axis=1)

# Convert False to 0
claimsDf.Labels = claimsDf.Labels.replace(1, 0)

# Convert True to 1
claimsDf.Labels = claimsDf.Labels.replace(5, 1)

"""Adding Evidence"""

def search(query, **kwargs):
    service = build("customsearch", "v1", developerKey=API_KEY)
    return service.cse().list(q=query, cx=CSE_ID, **kwargs).execute()

evidenceDf = pd.DataFrame(columns = [f"e{i+1}" for i in range(10)])

for claim in df.Text:
  lst = []
  try:
    results = search(claim, num=10)["items"]
  except:
    results = []
  for result in results:
    try:
      lst.append(result["title"])
    except:
      continue
  
  if len(lst) != 10:
    lst = [None for i in range(10)]

  evidenceDf.loc[len(evidenceDf)] = lst

"""Concatenate Results"""

finalDf = pd.concat([claimsDf, evidenceDf], axis=1)
finalDf.dropna(inplace=True)

# Swap dataframe column order
cols = finalDf.columns.tolist()
cols = cols[1:] + [cols[0]]
finalDf = finalDf[cols]

finalDf.to_csv("../data/dataset.csv", index=False)
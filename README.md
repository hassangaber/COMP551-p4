# COMP551-p4

## Introduction
* Reproducibility in ML project based on an Automatic Fake News Detection paper exploring if models are learning to reason: https://arxiv.org/pdf/2105.07698v1.pdf

* Paper repository: https://github.com/casperhansen/fake-news-reasoning

* Main claim: Current models classifying fake news based on both claims and evidence proves inferior to models based only on evidence. This highlights the issue that models are not learning to reason but rather exploit signals in evidence (bias)

## Dataset

* The paper uses the `Snopes` dataset and the `Politifact` dataset. Both of which can be found here: https://www.dropbox.com/s/3v5oy3eddg3506j/multi_fc_publicdata.zip?dl=0
* Data instances take on the form `D = {(c_1, e_1, y_1), ..., (c_n, e_n, y_n)}` where:
	- `c_j` is the claim in text; 
	- `e_j` is the evidence to support or refute the claim (evidence is collected from the top 10 google search results when using the claim as a query);
	- `y_j` is the target variable to be predicted 
* To fit the above data scheme the following features are excluded: 
	- For PolitiFact we exclude `[full flop, half flip, no flip]`
	- For Snopes we exclude `[unproven, miscaptioned, legend, outdated, misattributed, scam, correct attribution]`
* Datasets have a 70-10-20 split for training-validation-testing sets

## Results (F1 Macro Scores on Test Set)

| Model | Dataset | Input Type | F1 Macro Score |
|-------|---------|------------|----------------|
| RF    | Snopes  | Claim & Ev.|      0.281     |
|-------|---------|------------|----------------|
| RF	| Snopes  | Evidence   |		|
|-------|---------|------------|----------------| 		
| RF    | Snopes  | Claim      |	        |	
|-------|---------|------------|----------------|
| RF    | POTM    | Claim & Ev.|                |
|-------|---------|------------|----------------|
| RF	| POTM	  | Evidence   |	        |	
|-------|---------|------------|----------------|
| RF    | POTM    | Claim      |                |
|-------|---------|------------|----------------|
| BERT  |         |            |                |
|-------|---------|------------|----------------|
| BERT	|	  |	       |                |		
|-------|---------|------------|----------------|
| BERT	|	  |	       |                | 
|-------|---------|------------|----------------|
| LSTM	|	  |	       |                |
|-------|---------|------------|----------------|
| LSTM	|	  |            |                |
|-------|---------|------------|----------------|
| LSTM	|         |	       |	        |
|-------|---------|------------|----------------|
## Reproducing Results

### Re-running the Experiment
* Download the dataset and place it in the same file as `code-acl`
```
user@:~/fake-news-reasoning$ sudo wget https://www.dropbox.com/s/3v5oy3eddg3506j/multi_fc_publicdata.zip?dl=1
```

* Run `main.py`
```
user@:~/fake-news-reasoning/code-acl/bias$ sudo pip3 install -r requirement.txt
user@:~/fake-news-reasoning/code-acl/bias$ cp model_selection.py /usr/local/lib/python3.8/site-packages/hypopt
user@:~/fake-news-reasoning/code-acl/bias$ sudo python3 main.py --model lstm --lstm_layers 2 --lr 0.0001 --dataset snes   
```

### Interpreting Results
* After running each configuration between `[lstm, bert, rf]` and `[snes, pomt]`
```
user@:~/fake-news-reasoning/code-acl/bias$ cd results
user@:~/fake-news-reasoning/code-acl/bias/results$ ls && cd ..
user@:~/fake-news-reasoning/code-acl/bias$ sudo python3 analyze.py
```
 

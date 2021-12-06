# COMP551-p4

## Introduction
* Reproducibility in ML project based on an Automatic Fake News Detection paper exploring if models are learning to reason: https://arxiv.org/pdf/2105.07698v1.pdf
* Paper repository: https://github.com/casperhansen/fake-news-reasoning

* Main claim: Current models classifying fake news based on both claims and evidence proves inferior to models based only on evidence. This highlights the issue that models are not learning to reason but rather exploit signals in evidence (bias)

## Datasets

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

* The following table shows the results obtained from our independent run based on the paper repository and conducted experiments
* The paper claims that models do not learn better from claims and evidence as they do with evidence only, therefore, we focus on the claims & evidence and evidence only F1 scores
* There is an `experiments.sh` file in `src/` that runs all the following experiments


| Model | Dataset | Input Type | F1 Macro Score |
|-------|---------|------------|----------------|
| RF    | Snopes  | Claim & Ev.|      0.281     |
| RF	| Snopes  | Evidence   |      0.299     | 		
| RF    | Snopes  | Claim      |      0.225     |
| RF    | POTM    | Claim & Ev.|      0.306     |
| RF	| POTM	  | Evidence   |      0.297     |	
| RF    | POTM    | Claim      |      0.253     |
| BERT  | Snopes  | Claim & Ev.|                |
| BERT	| Snopes  |	Evidence   |                |
| LSTM	| POTM	  |	Claim & Ev.|                |
| LSTM	| POTM	  | Evidence   |                |
| LSTM	| Snopes  |	Claim & Ev.|	            |
| LSTM	| Snopes  |	Evidence   |	            |

## Reproducing Results Based on Project Paper

### Re-running the Experiment
* `setup.sh` will also do the following
* Clone the paper repository 
```
git clone https://github.com/casperhansen/fake-news-reasoning
```

* Download the dataset and place it in the same file as `code-acl`
```
cd fake-news-reasoning
sudo wget https://www.dropbox.com/s/3v5oy3eddg3506j/multi_fc_publicdata.zip?dl=1
```

* Run `main.py`
```
cd fake-news-reasoning/code-acl/bias
sudo pip3 install -r requirement.txt
cp model_selection.py /usr/local/lib/python3.8/site-packages/hypopt
sudo python3 main.py --model lstm --lstm_layers 2 --lr 0.0001 --dataset snes   
```

### Interpreting Results
* After running each configuration between `[lstm, bert, rf]` and `[snes, pomt]`, the following produces graphs in `results/`
```
cd results
ls && cd ..
sudo python3 analyze.py
```


# COMP551-p4

- `data` new dataset to be used in experiment
- `src` contains modified files from the paper repository and a script to automatically set up the enviroment
- `runs` contains a script to run our experiments and output files of runs
- `auto_fake_news_paper.pdf` is the paper to be reproduced
- All experiments in this project were executed on a google instance VM with an NVIDIA Tesla T4 GPU, CUDA 11.1

## Introduction
* Reproducibility in ML project based on an Automatic Fake News Detection paper exploring if models are learning to reason: https://arxiv.org/pdf/2105.07698v1.pdf
* Paper repository: https://github.com/casperhansen/fake-news-reasoning

* Main claim: Current models classifying fake news based on both claims and evidence proves inferior to models based only on evidence. This highlights the issue that models are not learning to reason but rather exploit signals in evidence (bias)

* Goal of this project is to obtain results from the paper on our own and apply models to other language classification datasets with identical dataset formats:
  * Observe how the metrics from the model with evidence only as input type consistently outpreform metrics from claim & evidence input types with constant computation times and model parameters

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

## Experiment I: Reproducing Results Based on Project Paper

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
* Install dependencies
```
cd fake-news-reasoning/code-acl/bias
sudo pip3 install -r requirement.txt
```
* Run `main.py`
```
cp model_selection.py /usr/local/lib/python3.8/site-packages/hypopt
sudo python3 main.py --model lstm --lstm_layers 2 --lr 0.0001 --dataset snes   
```

### Results (F1 Macro Scores on Test Set)

* The following table shows the results obtained from our independent run based on the paper repository
* The paper claims that models do not learn better from claims and evidence as they do with evidence only, therefore, we focus on the claims & evidence and evidence only F1 scores
* There is an `experiments.sh` file in `src/` that runs all the following experiments
* F1 macro score is logged as it provides a score for both the model's precision and recall ability

| Model | Dataset | Input Type | F1 Macro Score |
|-------|---------|------------|----------------|
| RF    | Snopes  | Claim & Ev.|      0.281     |
| RF	| Snopes  | Evidence   |      0.299     | 		
| RF    | Snopes  | Claim      |      0.225     |
| RF    | POTM    | Claim & Ev.|      0.306     |
| RF	| POTM	  | Evidence   |      0.297     |	
| RF    | POTM    | Claim      |      0.253     |
| BERT  | POTM    | Claim & Ev.|      0.335     |
| BERT	| POTM    |	Evidence   |      0.390     |
| LSTM	| POTM	  |	Claim & Ev.|      0.258     |
| LSTM	| POTM	  | Evidence   |      0.263     |
| LSTM	| Snopes  |	Claim & Ev.|	  0.259     |
| LSTM	| Snopes  |	Evidence   |	  0.266     |

### Synthesizing Graphs & Interpreting Results
* After running each configuration between `[lstm, bert, rf]` and `[snes, pomt]`, the following produces graphs in `results/`
```
cd results
ls && cd ..
sudo python3 analyze.py
```
* The results obtained in the above section align very well with the conclusions formed in the paper: evidence alone as an input type provides the model with better signals of fake news than models providing both claims and evidence
* This hints towards the fact that the models are not learning to reason from claims and evidence but rather are learning specific evidence patterns
* All evidence scores except for one outscored both the claim & evidence input types with the same models/datasets, the largest difference being the BERT model which learned much stronger signals from evidence alone

## Experiment II: Applying LSTM & Random Forrest models to new dataset

### Preprocessing for LSTM & RF
* The dataset was preprocessed to fit the format in the `Dataset` section above, all these operations are defined inside functions in `preprocess.py`
* The function `__ExportDataset__()` returns all the desired forms of the dataset: in torch.dataloader form and in an subscriptable dataset form for training, validation, and testing data
* Both the LSTM RNN and Random Forest only required the torch.dataloader generators for training, optimization, and metrics
* To stay consistent with the paper, we converted the dataset to tsv form like the original dataset
* To preprocess from scratch:
```
cd ~/fake-news-reasoning/code-acl
sudo wget https://www.dropbox.com/sh/29de7na30yrtjbx/AADAIzCP09SIdttdbX8QbJVSa?dl=1
unzip data.zip
cd /bias && python3 preprocess.py
```

### Results

#### Recall & Precision Metrics

| Model | Dataset | Input Type | F1 Macro Score |
|-------|---------|------------|----------------|
| LSTM  | Valid	  | Claim & Evi|      0.288     |
| LSTM  | Test	  | Claim & Evi|      0.252     |
| LSTM  | Valid	  | Evidence   |      0.290     |
| LSTM  | Test	  | Evidence   |      0.259     |
| RF    | Valid	  | Claim & Evi|      0.228     |
| RF    | Test	  | Claim & Evi|      0.213     |
| RF    | Valid	  | Evidence   |      0.282     |
| RF    | Test	  | Evidence   |      0.247     |

#### Computation Times

#### Model Parameters

### Remarks
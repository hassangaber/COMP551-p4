# COMP551-p4

## Introduction
* Reproducibility in ML project based on an Automatic Fake News Detection paper exploring if models are learning to reason: https://arxiv.org/pdf/2105.07698v1.pdf

* Paper repository: https://github.com/casperhansen/fake-news-reasoning

* Main claim: Current models classifying fake news based on both claims and evidence proves inferior to models based only on evidence. This highlights the issue that models are not learning to reason but rather exploit signals in evidence (bias)

## Dataset

* Data instances take on the form `D = {(c_1, e_1, y_1), ..., (c_n, e_n, y_n)}` where:
	- `c_j` is the claim in text; 
	- `e_j` is the evidence to support or refute the claim (evidence is collected from the top 10 google search results when using the claim as a query);
	- `y_j` is the target variable to be predicted 

# IMDb Sentiment Analysis
Machine learning methods to predict the positivity/sentiment of an IMDb movie review.

## Setup
1. Clone repository (run `git clone https://github.com/RylanSteinkey/imdbSentimentAnalysis.git`)
2. Change directories into the project folder: `cd imdbSentimentAnalysis`
3. [Download anaconda or miniconda (python 3.7)](https://conda.io/miniconda.html (python 3.7)), instructions for that [are here](https://conda.io/docs/user-guide/install/index.html)
4. Install dependecies: run `conda env create -f envi.yaml`
5. Run `snakemake`
6. Check results.txt for accuracy and a ranked list of important words


### Other Machine Learning Models

After the above has successfully run, you can execute other models by running models.py as:
`python models.py XGB` -- XGBoost                    (68.2% accuracy with 1000 samples)
`python models.py SVM` -- Support Vector Machine     (68.4% accuracy with 1000 samples)
`python models.py MNB` -- Multinomial Naive Bayes    (68.4% accuracy with 1000 samples)
`python models.py ANN` -- Artificial Neural Network  (78.8% accuracy with 1000 samples)

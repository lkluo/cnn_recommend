# CNN model for a recommendation project
In a nutshell, run bash.sh directly

# Dependency
- nltk package
- glove word2vec; here we choose **glove.840B.300d.txt**

## Data
- original data are stored in xlsx files
- data/tsv contains data extract from xlsx files

## Scripts
- `daataset.py`: prepare data for training/testing
- `data.py`: data preprocessing
- `model.py`: define CNN model
- `segmenter.py`: sentence segmentation
- `utils.py`: utility functions
- `main.py`: main script, including model training and prediction
- `metrics.py`: calculate metrics for prediction
- `run.sh`: run training & prediction

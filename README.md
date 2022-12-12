# Writeup Prize Winning Solution - Mars Spectrometry 2: Gas Chromatography

Username: jackson5

## Summary

My solution is an ensemble of ten identical Deep Neural Networks (DNNs) using tensorlflow. 
Each is trained using a combination of Label Distribution Learning (LDL) 
and a novel model averaging algorithm I name 'FixedSoup' inspired by ['GreedySoup'](https://arxiv.org/pdf/2203.05482.pdf) algorithm.
We construct the feature vectors using peak detection features alongside n-difference
features inspired by the derivatives produced by the [SG algorithm](https://pubs.acs.org/doi/10.1021/ac60214a047) in analytical chemistry. 
Indeed 'FixedSoup' improves the performance of a single model above the performance of an ensemble. 

# Setup

1. Create an environment using Python 3.8. The solution was originally run on Python 3.8.12. 
```
conda create --name example-submission python=3.8
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Download the data from the competition page into `data/raw`

The structure of the directory before running training or inference should be:
```
example_submission
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│       ├── submission_format.csv
│       ├── test_values.csv
│       ├── train_labels.csv
│       └── train_values.csv
├── models             <- Trained and serialized models, model predictions, or model summaries
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── make_dataset.py
│   ├── run_inference.py
│   ├── run_training.py
│   └── scorer.py
├── README.md          <- The top-level README for developers using this project.
├── requirements.txt   <- The requirements file for reproducing the analysis environment
├── Makefile           <- Makefile with commands like `make requirements`
└── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
```

# Hardware

The solution was run on macOS Monterey, version 12.1.
- Number of CPUs: 4
- Processor: 2 GHz Quad-Core Intel Core i5
- Memory: 16 GB 3733 MHz LPDDR4X

Both training and inference were run on CPU.
- Training time: ~2 minutes
- Inference time: ~1.5 minutes

# Run training

To run training from the command line: `python src/run_training.py`

```
$ python src/run_training.py --help
Usage: run_training.py [OPTIONS]

Options:
  --features-path PATH            Path to the raw training dataset for
                                  processing  [default:
                                  data/raw/train_values.csv]
  --labels-path PATH              Path to the training labels  [default:
                                  data/raw/train_labels.csv]
  --model-save-path PATH          Path to save the trained model weights
                                  [default: models/random_forest.pkl]
  --debug / --no-debug            Run on a small subset of the data for
                                  debugging  [default: no-debug]
  --help                          Show this message and exit.
```

By default, trained model weights will be saved to `models/random_forest.pkl`. The model weights file that is saved out is ~25 MB.

Trained model weights can be downloaded from this Google folder: https://drive.google.com/drive/folders/1LW3PjXh_rL49VuEwj6oK5PuRM5nD5rPj?usp=sharing

You can use `wget` to download the model weights programmatically:
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bmE9oVgi58deRdYicfBzNzNButfzjcjs' -O models/random_forest.pkl
```

# Run inference

To run inference from the command line: `python src/run_inference.py`

```
$ python src/run_inference.py --help
Usage: run_inference.py [OPTIONS]

Options:
  --model-path PATH               Path to the saved model weights  [default:
                                  models/random_forest.pkl]
  --features-path PATH            Path to the test features  [default:
                                  data/raw/test_values.csv]
  --submission-save-path PATH     Path to save the generated submission
                                  [default: data/processed/submission.csv]
  --submission-format-path PATH   Path to the submission format csv  [default:
                                  data/raw/submission_format.csv]
  --debug / --no-debug            Run on a small subset of the data for
                                  debugging  [default: no-debug]
  --help                          Show this message and exit.
```

By default, predictions will be saved out to `data/processed/submission.csv`.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
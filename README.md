# IMDB Sentiment Analysis

*Part of Project 2 of COMP 551 Applied Machine Learning - McGill University*  
*Members: Le Nhat Hung, Han Wen Xie, Michel Abdelnour*

## Prerequisites

### Running Google Colab locally *(optional)*

1. Install and enable the jupyter_http_over_ws jupyter extension (one-time)
    ```
    pip install jupyter_http_over_ws
    jupyter serverextension enable --py jupyter_http_over_ws
    ```

2. Start server and authenticate
    ```
    jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
    ```

## Project Tasks

The timeline of project tasks, experimentation and testing are outlined in our [Colab](https://colab.research.google.com/drive/1oT0mCFWeHc8mHC20V8X12tX_NeNgF3_4).

## Usage

A Makefile with commands is included to make running scripts easier.  
**Make sure the project's structure is kept as follow:**

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    │                         generated with `pip freeze > requirements.txt`
    │
    └── code                <- Source code for use in this project.
        ├── __init__.py    <- Makes code a Python module
        ├── data.py           <- Scripts to preprocess raw data
        ├── models.py       <- Code for models (Bernoulli Naive Bayes)
        ├── train.py       <- Train models
        └── predict.py         <-

1. Install required packages: `make requirements`
2. Build datasets: `make data`
3. Train models: `make train`
4. Make predictions: `make predict`

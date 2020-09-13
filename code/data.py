# -*- coding: utf-8 -*-
import logging
import json
from pathlib import Path
from zipfile import ZipFile

project_dir = Path(__file__).resolve().parents[1]
raw_path = project_dir / 'data/raw'
processed_path = project_dir / 'data/processed'

NEGATIVE, POSITIVE = 0, 1

def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final datasets from raw data')
    preprocess(raw_path, processed_path) # Group all data into json datasets

def preprocess(input_path, output_path):
    files = [ 'train.zip', 'test.zip' ]

    for file in files:
        set_name = file.split('.')[0]
        X, y = [], []

        with ZipFile(input_path / file) as f:
            txtfiles = (
                name
                for name in f.namelist()
                if '.txt' in name and 'MACOSX' not in name # Ignore 'MACOSX' directory
            )

            # Manually rearranging test data in order for Kaggle submission
            if set_name == 'test':
                txtfiles = sorted(txtfiles,
                    key = lambda filename : int(filename
                        .split('/')[-1]
                            .split('.')[0]))

            for txtfile in txtfiles:
                text = f.read(txtfile).decode('utf-8')
                X.append(text)

                # Add sentiment label if in training dataset
                if set_name == 'train':
                    y.append(POSITIVE if 'pos' in txtfile else NEGATIVE)

        for data, prefix in zip([X, y], ['X_', 'y_']):
            if not (set_name == 'test' and prefix == 'y_'):
                with open(output_path / (prefix + set_name + '.json'), 'w') as fout:
                    json.dump(data, fout)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

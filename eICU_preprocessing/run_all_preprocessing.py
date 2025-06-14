from eICU_preprocessing.timeseries import timeseries_main
from eICU_preprocessing.diagnoses import diagnoses_main
from eICU_preprocessing.flat_and_labels import flat_and_labels_main
from eICU_preprocessing.split_train_test import split_train_test
import argparse
import os
import json

with open('paths.json', 'r') as f:
    eICU_path = json.load(f)["eICU_path"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run all eICU preprocessing steps')
    parser.add_argument(
        '--full', action='store_true',
        help='process the entire dataset instead of the small test subset')
    args = parser.parse_args()

    is_test = not args.full

    print('==> Removing the stays.txt file if it exists...')
    try:
        os.remove(eICU_path + 'stays.txt')
    except FileNotFoundError:
        pass

    cut_off_prevalence = 0.01  # this would be 1%
    timeseries_main(eICU_path, test=is_test)
    split_train_test(eICU_path, is_test=is_test)

    diagnoses_main(eICU_path, cut_off_prevalence)
    flat_and_labels_main(eICU_path)
    split_train_test(eICU_path, is_test=True)

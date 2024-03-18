


import sys
import os
import argparse
import pandas as pd
import logging

from utils import *

def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--input-data-folder", type=str, required=False)
    parser.add_argument("--from-pretrained", type=bool, required=False)
    parser.add_argument("--architecture", type=str, required=False)
    parser.add_argument("--data-augmentation", type=bool, required=False)

    args = parser.parse_args()


    # check if the input data folder is provided
    if args.input_data_folder is None:
        # No need to reload the data
        train_folder, val_folder = TRAIN_FOLDER, VAL_FOLDER
    else:
        prepare_folders()

        # create the csv file
        data_file = create_csv_file(args.input_data_folder)

        train_folder, val_folder = load_data_into_folders(data_file)


    model_path = train_model(train_folder, val_folder, architecture=args.architecture,
                             from_pretrained=args.from_pretrained, epochs=3, data_augmentation=args.data_augmentation)
    # results = evaluate_model(model_path, val_folder)





    



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()




import sys
import os
import argparse
import pandas as pd
import logging

from utils import *

def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--input_data_folder", type=str, required=True)
    parser.add_argument("--from_pretrained", type=str, required=False)

    args = parser.parse_args()

    # create the csv file
    data_file = create_csv_file(args.input_data_folder)

    train_folder, val_folder = load_data_into_folders(data_file)


    model_path = train_model(train_folder, val_folder, from_pretrained=args.from_pretrained)

    results = evaluate_model(model, val_folder)





    



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

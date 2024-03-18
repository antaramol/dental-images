


import argparse
import logging
import time

from utils import *

def main():
    start = time.time()
    # parse arguments
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--input-data-folder", type=str, required=False)
    parser.add_argument("--architecture", type=str, required=False)

    parser.add_argument("--from-pretrained", action="store_true", required=False)
    parser.add_argument("--data-augmentation", action="store_true", required=False)

    args = parser.parse_args()

    logging.info(args)


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
                             from_pretrained=args.from_pretrained, epochs=60, data_augmentation=args.data_augmentation)
    # results = evaluate_model(model_path, val_folder)

    # log the time it took to run the pipeline in minutes
    elapsed = (time.time() - start) / 60
    logging.info(f"Elapsed time: {elapsed:.2f} minutes")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

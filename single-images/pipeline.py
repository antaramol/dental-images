


import argparse
import logging
import time
import os
import numpy as np
import pandas as pd

from utils import *

def main():
    start = time.time()
    # parse arguments
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--input-data-folder", type=str, required=False)
    parser.add_argument("--architecture", type=str, required=False)

    parser.add_argument("--from-pretrained", action="store_true", required=False)
    parser.add_argument("--data-augmentation", action="store_true", required=False)
    parser.add_argument("--fixed-feature-extractor", action="store_true", required=False)
    parser.add_argument("--k-fold", type=int, required=False)
    parser.add_argument("--weights", type=str, required=False)


    args = parser.parse_args()

    logging.info(args)

    k = args.k_fold if args.k_fold is not None else 5


    # check if the input data folder is provided
    if args.input_data_folder is None:
        # No need to reload the data
        train_folder, val_folder = TRAIN_FOLDER, VAL_FOLDER
    else:
        prepare_folders()

        # create the csv file
        data_file = create_csv_file(args.input_data_folder)


        if args.k_fold is not None:
            k_fold_folders = load_data_into_k_fold_folders(data_file, k=args.k_fold)
        else:
            train_folder, val_folder = load_data_into_folders(data_file)

    # check if the k-fold folders variable is defined
    # if "k_fold_folders" not in locals():
    #     k_fold_folders = [os.path.join(DATASET_FOLDER, f"k_fold_{i}") for i in range(args.k_fold)]



    if args.k_fold is not None:
        
        mean_accuracy = 0
        model_predictions = []
        real_labels = []

        # train the model on the k-fold folders (train and val are subfolders inside each k-fold folder)
        for k_fold_folder in k_fold_folders:
            logging.info(f"Training model on {k_fold_folder}")

            train_folder = os.path.join(k_fold_folder, "train")
            val_folder = os.path.join(k_fold_folder, "val")

            dataloaders, dataset_sizes, class_names, device = load_dataset(train_folder, val_folder, args.data_augmentation)


            model_path = train_model(dataloaders, dataset_sizes, class_names, device,
                                     architecture=args.architecture, weights=args.weights,
                                     from_pretrained=args.from_pretrained, epochs=100, data_augmentation=args.data_augmentation, fixed_feature_extractor=args.fixed_feature_extractor)
            logging.info(f"Model path: {model_path}")

            accuracy, predictions, labels = evaluate_model(model_path, dataloaders, device)

            mean_accuracy += accuracy
            model_predictions.append(predictions)

            real_labels.append(labels)

        mean_accuracy /= args.k_fold


        # log FINAL RESULTS to the console

        logging.info("")
        logging.info("FINAL RESULTS")

        logging.info(f"Mean accuracy: {mean_accuracy}")

        # compare the predictions with the real labels
        model_predictions = np.concatenate(model_predictions)
        real_labels = np.concatenate(real_labels)

        accuracy = np.mean(model_predictions == real_labels)
        logging.info(f"Final accuracy: {accuracy}")
            

        # save the mean accuracy to a csv file in the output folder, with the architecture, from_pretrained, data_augmentation, and fixed_feature_extractor as columns
        
        k_fold_results = pd.DataFrame({"architecture": [args.architecture], "from_pretrained": [args.from_pretrained], "data_augmentation": [args.data_augmentation], "fixed_feature_extractor": [args.fixed_feature_extractor], "mean_accuracy": [mean_accuracy], "final_accuracy": [accuracy]})

        if os.path.exists(os.path.join(OUTPUTS_FOLDER, "k_fold_results.csv")):
            old_k_fold_results = pd.read_csv(os.path.join(OUTPUTS_FOLDER, "k_fold_results.csv"))
            k_fold_results = pd.concat([old_k_fold_results, k_fold_results], ignore_index=True)

        k_fold_results.to_csv(os.path.join(OUTPUTS_FOLDER, "k_fold_results.csv"), index=False)

    else:

        # dataloaders, dataset_sizes, class_names, device = load_dataset(train_folder, val_folder, args.data_augmentation)

        # train the model on the train folder
        try:
            model_path = train_model(DATASET_FOLDER,
                                 architecture=args.architecture, weights=args.weights,
                                 from_pretrained=args.from_pretrained, epochs=60, data_augmentation=args.data_augmentation, fixed_feature_extractor=args.fixed_feature_extractor)
        except Exception as e:
            logging.error(f"{e}")
            return
        logging.info(f"Model path: {model_path}")


    # log the time it took to run the pipeline in minutes
    elapsed = (time.time() - start) / 60
    logging.info(f"Elapsed time: {elapsed:.2f} minutes")


    logging.info("")



if __name__ == "__main__":
    # log into a file
    logging.basicConfig(filename='pipeline.log', level=logging.INFO)
    # logging.basicConfig(level=logging.INFO)

    main()

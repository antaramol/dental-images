


import argparse
import logging
import time
import os
import numpy as np
import pandas as pd

# import softmax function
from scipy.special import softmax

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
    parser.add_argument("--learning-rate", type=float, required=False)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--epochs", type=int, required=False)

    parser.add_argument("--score-level-fusion", type=int, required=False)
    parser.add_argument("--k-fold", type=int, required=False)
    parser.add_argument("--weights", type=str, required=False)


    args = parser.parse_args()

    logging.info(args)

    k = args.k_fold if args.k_fold is not None else 5

    n = args.score_level_fusion if args.score_level_fusion is not None else 1


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
        
        # mean_accuracy = 0
        k_fold_accuracies = []
        model_predictions = []
        real_labels = []
        test_subjects = []

        # train the model on the k-fold folders (train and val are subfolders inside each k-fold folder)
        for k_fold_folder in k_fold_folders:
            logging.info(f"Training model on {k_fold_folder}")

            train_folder = os.path.join(k_fold_folder, "train")
            val_folder = os.path.join(k_fold_folder, "val")

            last_layer_weights = []

            for i in range(n):

                dataloaders, dataset_sizes, class_names, device = load_dataset(k_fold_folder,
                                                                            args.data_augmentation, args.batch_size)


                model_path, history = train_model(dataloaders, dataset_sizes, class_names, device,
                                    architecture=args.architecture, weights=args.weights,
                                    from_pretrained=args.from_pretrained, epochs=args.epochs, learning_rate=args.learning_rate,
                                    fixed_feature_extractor=args.fixed_feature_extractor)            
                

                # evaluate the model on the val folder
                accuracy, predictions, labels, pred_weights = evaluate_model(model_path, dataloaders, device)

                last_layer_weights.append(pred_weights)

                logging.info(f"Accuracy: {accuracy}")

                # update the results csv
                update_results_csv(args.architecture, args.from_pretrained, args.weights,
                                    args.fixed_feature_extractor, args.data_augmentation, args.epochs, args.learning_rate,
                                    args.batch_size, max(history['val']['acc']), accuracy, model_path)

                
                avg_weights = []
                max_weights = []
                min_weights = []
                # sum last_layer_weights to get the final weights
                if i == args.score_level_fusion - 1:
                    for j in range(len(last_layer_weights)):
                        # apply the softmax function to the weights
                        last_layer_weights[j] = softmax(last_layer_weights[j], axis=1)
                        # sum the weights
                        if j == 0:
                            avg_weights = last_layer_weights[j]
                            max_weights = last_layer_weights[j]
                            min_weights = last_layer_weights[j]
                        else:
                            avg_weights = np.add(avg_weights, last_layer_weights[j])
                            max_weights = np.maximum(max_weights, last_layer_weights[j])
                            min_weights = np.minimum(min_weights, last_layer_weights[j])


                    avg_predictions = np.argmax(avg_weights, axis=1)
                    max_predictions = np.argmax(max_weights, axis=1)
                    min_predictions = np.argmax(min_weights, axis=1)
                    

                    avg_score_level_fusion_accuracy = np.mean(avg_predictions == labels)
                    max_score_level_fusion_accuracy = np.mean(max_predictions == labels)
                    min_score_level_fusion_accuracy = np.mean(min_predictions == labels)

                    # update the results csv
                    score_level_fusion_results = pd.DataFrame({"architecture": [args.architecture], "from_pretrained": [args.from_pretrained], "data_augmentation": [args.data_augmentation], "fixed_feature_extractor": [args.fixed_feature_extractor], "epochs": [args.epochs], "learning_rate": [args.learning_rate], "batch_size": [args.batch_size], "score_level_fusion": [args.score_level_fusion]
                    , "avg_score_level_fusion_accuracy": [avg_score_level_fusion_accuracy], "max_score_level_fusion_accuracy": [max_score_level_fusion_accuracy], "min_score_level_fusion_accuracy": [min_score_level_fusion_accuracy]})

                    if os.path.exists(os.path.join(OUTPUTS_FOLDER, "score_level_fusion_results.csv")):
                        old_score_level_fusion_results = pd.read_csv(os.path.join(OUTPUTS_FOLDER, "score_level_fusion_results.csv"))
                        score_level_fusion_results = pd.concat([old_score_level_fusion_results, score_level_fusion_results], ignore_index=True)

                    score_level_fusion_results.to_csv(os.path.join(OUTPUTS_FOLDER, "score_level_fusion_results.csv"), index=False)




            
            k_fold_accuracies.append(accuracy)


            model_predictions.append(predictions)
            real_labels.append(labels)
            
            # image name is in the string between the first underscore and the last dot
            
            subject_ids = [os.path.basename(str(image_path)).split("soggetto_")[1].split(".")[0]
                           for image_path in dataloaders["val"].dataset.samples]
            
            test_subjects.append(subject_ids)


        mean_accuracy = np.mean([accuracy for accuracy in k_fold_accuracies])

        std_accuracy = np.std([accuracy for accuracy in k_fold_accuracies])


        # log FINAL RESULTS to the console

        logging.info("")
        logging.info("FINAL RESULTS")

        logging.info(f"Mean accuracy: {mean_accuracy}")

        logging.info(f"Standard deviation: {std_accuracy}")


        # save the mean, std accuracy to a csv file in the output folder, with the architecture, from_pretrained, data_augmentation, and fixed_feature_extractor as columns
        
        k_fold_results = pd.DataFrame({"architecture": [args.architecture], "from_pretrained": [args.from_pretrained], "data_augmentation": [args.data_augmentation], "fixed_feature_extractor": [args.fixed_feature_extractor], "mean_accuracy": [mean_accuracy], "std_accuracy": [std_accuracy]})

        if os.path.exists(os.path.join(OUTPUTS_FOLDER, "k_fold_results.csv")):
            old_k_fold_results = pd.read_csv(os.path.join(OUTPUTS_FOLDER, "k_fold_results.csv"))
            k_fold_results = pd.concat([old_k_fold_results, k_fold_results], ignore_index=True)

        k_fold_results.to_csv(os.path.join(OUTPUTS_FOLDER, "k_fold_results.csv"), index=False)



        # compare the predictions with the real labels
        model_predictions = np.concatenate(model_predictions)
        real_labels = np.concatenate(real_labels)

        final_accuracy = np.mean(model_predictions == real_labels)
        logging.info(f"Final accuracy: {final_accuracy}")


        # save the predictions and real labels to a csv file in the output folder, with subjet_id, age, prediction, and real_label as columns
        predictions_df = pd.DataFrame({"subject_id": np.concatenate(test_subjects), "prediction": model_predictions, "real_label": real_labels})

        # save the predictions to a csv file, if the file already exists, delete it and create a new one
        predictions_file = os.path.join(OUTPUTS_FOLDER, "predictions.csv")
        if os.path.exists(predictions_file):
            os.remove(predictions_file)

        predictions_df.to_csv(predictions_file, index=False)

    else:

        last_layer_weights = []

        for i in range(n):
    

            dataloaders, dataset_sizes, class_names, device = load_dataset(DATASET_FOLDER, args.data_augmentation, args.batch_size)
            # train the model on the train folder
            try:
                model_path, history = train_model(dataloaders, dataset_sizes, class_names, device,
                                    architecture=args.architecture, weights=args.weights,
                                    from_pretrained=args.from_pretrained, epochs=args.epochs, learning_rate=args.learning_rate,
                                    fixed_feature_extractor=args.fixed_feature_extractor)
                


                # evaluate the model on the val folder
                accuracy, predictions, labels, pred_weights = evaluate_model(model_path, dataloaders, device)

                last_layer_weights.append(pred_weights)

                logging.info(f"Accuracy: {accuracy}")

                # update the results csv
                update_results_csv(args.architecture, args.from_pretrained, args.weights,
                                    args.fixed_feature_extractor, args.data_augmentation, args.epochs, args.learning_rate,
                                    args.batch_size, max(history['val']['acc']), accuracy, model_path)
                
            except Exception as e:
                logging.error(f"{e}")
                return
            logging.info(f"Model path: {model_path}")


            avg_weights = []
            max_weights = []
            min_weights = []
            # sum last_layer_weights to get the final weights
            if i == args.score_level_fusion - 1:
                for j in range(len(last_layer_weights)):
                    # apply the softmax function to the weights
                    last_layer_weights[j] = softmax(last_layer_weights[j], axis=1)
                    # sum the weights
                    if j == 0:
                        avg_weights = last_layer_weights[j]
                        max_weights = last_layer_weights[j]
                        min_weights = last_layer_weights[j]
                    else:
                        avg_weights = np.add(avg_weights, last_layer_weights[j])
                        max_weights = np.maximum(max_weights, last_layer_weights[j])
                        min_weights = np.minimum(min_weights, last_layer_weights[j])


                avg_predictions = np.argmax(avg_weights, axis=1)
                max_predictions = np.argmax(max_weights, axis=1)
                min_predictions = np.argmax(min_weights, axis=1)
                

                avg_score_level_fusion_accuracy = np.mean(avg_predictions == labels)
                max_score_level_fusion_accuracy = np.mean(max_predictions == labels)
                min_score_level_fusion_accuracy = np.mean(min_predictions == labels)

                # update the results csv
                score_level_fusion_results = pd.DataFrame({"architecture": [args.architecture], "from_pretrained": [args.from_pretrained], "data_augmentation": [args.data_augmentation], "fixed_feature_extractor": [args.fixed_feature_extractor], "epochs": [args.epochs], "learning_rate": [args.learning_rate], "batch_size": [args.batch_size], "score_level_fusion": [args.score_level_fusion]
                , "avg_score_level_fusion_accuracy": [avg_score_level_fusion_accuracy], "max_score_level_fusion_accuracy": [max_score_level_fusion_accuracy], "min_score_level_fusion_accuracy": [min_score_level_fusion_accuracy]})

                if os.path.exists(os.path.join(OUTPUTS_FOLDER, "score_level_fusion_results.csv")):
                    old_score_level_fusion_results = pd.read_csv(os.path.join(OUTPUTS_FOLDER, "score_level_fusion_results.csv"))
                    score_level_fusion_results = pd.concat([old_score_level_fusion_results, score_level_fusion_results], ignore_index=True)

                score_level_fusion_results.to_csv(os.path.join(OUTPUTS_FOLDER, "score_level_fusion_results.csv"), index=False)

            


    # log the time it took to run the pipeline in minutes
    elapsed = (time.time() - start) / 60
    logging.info(f"Elapsed time: {elapsed:.2f} minutes")


    logging.info("")



if __name__ == "__main__":
    # log into a file
    logging.basicConfig(filename='pipeline.log', level=logging.INFO)
    # logging.basicConfig(level=logging.INFO)

    main()

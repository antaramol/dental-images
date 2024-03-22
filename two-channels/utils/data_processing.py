
import pandas as pd
import os
import glob
import logging
import shutil
import numpy as np
import cv2
from PIL import Image

OUTPUTS_FOLDER = "outputs_2d"
DATASET_FOLDER = OUTPUTS_FOLDER + "/data"
MODELS_FOLDER = OUTPUTS_FOLDER + "/models"

TRAIN_FOLDER = DATASET_FOLDER + "/train"
VAL_FOLDER = DATASET_FOLDER + "/val"

CLASSES = ["under_18", "over_18"]

SEED = 42



def prepare_folders():

    # create the outputs, data and models folders if they don't exist
    for folder in [OUTPUTS_FOLDER, DATASET_FOLDER, MODELS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    
    # clear data folder
    os.system(f"rm -rf {DATASET_FOLDER}/*")

    # create train and val folders
    for folder in [TRAIN_FOLDER, VAL_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
                
            for c in CLASSES:
                class_folder = os.path.join(folder, c)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
            


def create_csv_file(input_data_folder):
    data = pd.DataFrame(columns=["subject_id", "age", "image", "side", "tooth_present", "under_18"])

    for file_name in glob.glob(os.path.join(input_data_folder, "*.csv")):
        subject = pd.read_csv(file_name, header=None)
        # subject id is the first field of the subject
        subject_id = subject.iloc[0, 0]
        # age is the 10th field of the subject
        age = subject.iloc[0, 9]

        # left image is soggetto_SUBJECTID_sx.bmp
        left_image = os.path.join(input_data_folder, f"soggetto_{subject_id}_sx.bmp")
        # right image is soggetto_SUBJECTID_dx.bmp
        right_image = os.path.join(input_data_folder, f"soggetto_{subject_id}_dx.bmp")

        # left tooth present is fifth field of the subject
        left_tooth_present = [True if subject.iloc[0, 4] == "dente presente" else False][0]
        # right tooth present is second field of the subject
        right_tooth_present = [True if subject.iloc[0, 1] == "dente presente" else False][0]


        data = pd.concat([data, pd.DataFrame({"subject_id": [subject_id], "age": [age], "image": [left_image], "side": ["left"], "tooth_present": [left_tooth_present]})])
        data = pd.concat([data, pd.DataFrame({"subject_id": [subject_id], "age": [age], "image": [right_image], "side": ["right"], "tooth_present": [right_tooth_present]})])

    # order the dataframe by subject_id and reset the index
    data = data.sort_values("subject_id").reset_index(drop=True)

    # add under_18 column based on age
    data["under_18"] = data["age"] < 18

    output_file = os.path.join(DATASET_FOLDER, "dental-data.csv")


    # save to csv
    data.to_csv(output_file, index=False)

    return output_file


def load_data_into_folders(data_file):


    data = pd.read_csv(data_file)

    # save pictures into train and val folders, 80% train, 20% val
    # each folder should have a subfolder for each class (under 18 and over 18)

    unique_subjects = data["subject_id"].unique()
    np.random.seed(SEED)

    # shuffle the subjects
    np.random.shuffle(unique_subjects)

    # 80% of the subjects will be used for training
    subjects = {'train': unique_subjects[:int(0.8 * len(unique_subjects))], 'val': unique_subjects[int(0.8 * len(unique_subjects)):]}

    for subject_type, subject_list in subjects.items():
        for subject in subject_list:
            subject_data = data[data["subject_id"] == subject]
            # stack the left and right images (grayscale) from 2x 2D images to 1x 3D image
            images = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in subject_data["image"]]

            class_folder = os.path.join(DATASET_FOLDER, subject_type, "under_18" if subject_data.iloc[0]["under_18"] else "over_18")
            
            stacked_image = np.stack(images, axis=-1)
            # add dimension of zeros to the stacked image
            stacked_image = np.concatenate([stacked_image, np.zeros_like(images[0])[..., np.newaxis]], axis=-1)
            # save the stacked image as lossless rgb bmp
            im = Image.fromarray(stacked_image)
            im.save(os.path.join(class_folder, f"{subject}.bmp"))





    return TRAIN_FOLDER, VAL_FOLDER



def load_data_into_k_fold_folders(data_file, k=5):
    
        data = pd.read_csv(data_file)
    
        # shuffle the data
        data = data.sample(frac=1, random_state=SEED)
    
        # split the data into k folds
        k_fold_data = np.array_split(data, k)
    
        # save data into k-fold folders, each folder contains a train and val folder with subfolders for each class
        k_fold_folders = []
        
        for i in range(k):
            k_fold_folder = os.path.join(DATASET_FOLDER, f"k_fold_{i}")
            k_fold_folders.append(k_fold_folder)
            
            for folder in [k_fold_folder + "/train", k_fold_folder + "/val"]:
                for c in CLASSES:
                    class_folder = os.path.join(folder, c)
                    if not os.path.exists(class_folder):
                        os.makedirs(class_folder)
            
            # copy the images to the train and val folders
            for j, df in enumerate(k_fold_data):
                folder = k_fold_folder + "/val" if j == i else k_fold_folder + "/train"
                for index, row in df.iterrows():
                    image_path = row["image"]
                    class_folder = os.path.join(folder, "under_18" if row["under_18"] else "over_18")
                    shutil.copy(image_path, class_folder)

        return k_fold_folders
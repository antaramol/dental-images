
import pandas as pd
import os
import glob
import logging
import shutil

OUTPUTS_FOLDER = "outputs"
DATASET_FOLDER = OUTPUTS_FOLDER + "/data"
MODELS_FOLDER = OUTPUTS_FOLDER + "/models"

TRAIN_FOLDER = DATASET_FOLDER + "/train"
VAL_FOLDER = DATASET_FOLDER + "/val"

CLASSES = ["under_18", "over_18"]

SEED = 42



def prepare_folders():

    # create the outputs, data and models folders if they don't exist
    for folder in [OUTPUTS_FOLDER, DATASET_FOLDER, MODELS_FOLDER, TRAIN_FOLDER, VAL_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    
    # clear data folders
    for folder in [TRAIN_FOLDER, VAL_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # create the under_18 and over_18 folders
    for folder in [TRAIN_FOLDER, VAL_FOLDER]:
        for c in CLASSES:
            class_folder = os.path.join(folder, c)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

    # delete csv file if it exists already
    data_file = os.path.join(DATASET_FOLDER, "dental-data.csv")
    if os.path.exists(data_file):
        os.remove(data_file)
            


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

    # shuffle the data
    data = data.sample(frac=1, random_state=SEED)

    # split the data into train and val
    train_data = data.sample(frac=0.8, random_state=SEED)
    val_data = data.drop(train_data.index)

    # copy the images to the train and val folders
    for folder, df in zip([TRAIN_FOLDER, VAL_FOLDER], [train_data, val_data]):
        for c in CLASSES:
            class_folder = os.path.join(folder, c)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

        for index, row in df.iterrows():
            image_path = row["image"]
            class_folder = os.path.join(folder, "under_18" if row["under_18"] else "over_18")
            shutil.copy(image_path, class_folder)

    return TRAIN_FOLDER, VAL_FOLDER
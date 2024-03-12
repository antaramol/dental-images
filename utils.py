
#%%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(columns=["subject_id", "age", "image", "side"])

data_path = "data/dental-images/UP_DOWN_stadiazione_CH_gimp"

for file_name in glob.glob(os.path.join(data_path, "*.csv")):
    subject = pd.read_csv(file_name, header=None)
    # subject id is the first field of the subject
    subject_id = subject.iloc[0, 0]
    # age is the 10th field of the subject
    age = subject.iloc[0, 9]

    # left image is soggetto_SUBJECTID_sx.bmp
    left_image = os.path.join(data_path, f"soggetto_{subject_id}_sx.bmp")
    # right image is soggetto_SUBJECTID_dx.bmp
    right_image = os.path.join(data_path, f"soggetto_{subject_id}_dx.bmp")

    data = pd.concat([data, pd.DataFrame({"subject_id": [subject_id], "age": [age], "image": [left_image], "side": ["left"]})])
    data = pd.concat([data, pd.DataFrame({"subject_id": [subject_id], "age": [age], "image": [right_image], "side": ["right"]})])



data.head()


# %%
# order the dataframe by subject_id and reset the index
data = data.sort_values("subject_id").reset_index(drop=True)


# add under_18 column based on age
data["under_18"] = data["age"] < 18

# save to csv
data.to_csv("data/dental-images/dental-data.csv", index=False)


data.head()

# %%

data.describe()
# %%

print(f"Percentage of under 18: {data['under_18'].value_counts(normalize=True)[True] * 100:.2f}%")
data["under_18"].value_counts()

# %%
# plot the age distribution, different colors for under 18 and over 18
plt.hist(data[data["under_18"]]["age"], bins=20, alpha=0.5, label="under 18")
plt.hist(data[~data["under_18"]]["age"], bins=20, alpha=0.5, label="over 18")
plt.legend()
plt.show()

# plot data count for each class
data["under_18"].value_counts().plot(kind="bar")
plt.show()


#%%
# leave N subjects out for testing (5 random subjects_id)
N = 10
seed = 42
test_subjects = data["subject_id"].sample(N, random_state=seed)
test_data = data[data["subject_id"].isin(test_subjects)]
train_val_data = data[~data["subject_id"].isin(test_subjects)]



# %%

# save pictures into train and val folders, 80% train, 20% val
# each folder should have a subfolder for each class (under 18 and over 18)
import shutil

train_path = "data/dental-images/train"
val_path = "data/dental-images/val"
test_path = "data/dental-images/test"

# create the folders if they don't exist
for folder in [train_path, val_path, test_path]:
    if not os.path.exists(folder):
        os.makedirs(os.path.join(folder, "under_18"))
        os.makedirs(os.path.join(folder, "over_18"))


# clear the folders
for folder in [train_path, val_path, test_path]:
    for subfolder in ["under_18", "over_18"]:
        for file in glob.glob(os.path.join(folder, subfolder, "*")):
            os.remove(file)

#%%
# shuffle the training data
train_val_data = train_val_data.sample(frac=1, random_state=seed)


#%%
# 80% of the data is for training
train_data = train_val_data.iloc[:int(0.8 * len(train_val_data))]
# 20% of the data is for validation
val_data = train_val_data.iloc[int(0.8 * len(train_val_data)):]

# move the images to the folders
for i, row in train_data.iterrows():
    shutil.copy(row["image"], os.path.join(train_path, "under_18" if row["under_18"] else "over_18"))

for i, row in val_data.iterrows():
    shutil.copy(row["image"], os.path.join(val_path, "under_18" if row["under_18"] else "over_18"))

# save the test data
for i, row in test_data.iterrows():
    shutil.copy(row["image"], os.path.join(test_path, "under_18" if row["under_18"] else "over_18"))


# %%
        
# show histograms of the number of images in each class
        
classes = ["over_18", "under_18"]

train_counts = train_data["under_18"].value_counts()
val_counts = val_data["under_18"].value_counts()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].bar(classes, train_counts)
ax[0].set_title("Train")
ax[1].bar(classes, val_counts)
ax[1].set_title("Validation")
plt.show()

# %%
# show histogram of the number of images in each class
under_18_counts = data["under_18"].value_counts()
plt.bar(classes, under_18_counts)
plt.title("All data")
plt.show()


# %%
print(f"Total images: {len(data)}")
print(f"Train images: {len(train_data)}")
print(f"Validation images: {len(val_data)}")
print(f"Test images: {len(test_data)}")
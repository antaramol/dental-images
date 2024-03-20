    
#%%
# read outputs_single_images_weights/results.csv
import pandas as pd

results = pd.read_csv("outputs_single_images_weights/results.csv")
results
# %%

architectures = results["architecture"].unique()
architectures

# %%
# read sorted_models.json

import json

with open("sorted_models.json", "r") as f:
    sorted_models = json.load(f)

sorted_models

# %%
model_names = [model[0] for model in sorted_models]

# remove IMAGENET1k and tail from model names
model_names = [model.split("_IMAGENET1K")[0] for model in model_names]
model_names
# %%

# compare model names with architectures
for model in model_names:
    if model not in architectures:
        print(model)
        


# %%

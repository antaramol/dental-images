    
#%%
# read outputs_single_images_weights/results.csv
import pandas as pd

results = pd.read_csv("outputs_single_images_weights/results.csv")
results
# %%

architectures = results["architecture"].unique()
architectures

# %%

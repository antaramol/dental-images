#%%
import torch

from torchvision import models

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
model



#%%

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model

# %%

# load one image from the dataset
from torchvision.transforms import v2

from PIL import Image

img = Image.open('UP_DOWN_stadiazione_CH_gimp/soggetto_1_dx.bmp')

# plot the image
import matplotlib.pyplot as plt

plt.imshow(img)
plt.show()

# %%
# run the image through the model to get the prediction

transform = v2.Compose([
            v2.Resize(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

img = transform(img)

img = img.unsqueeze(0)

model.eval()
output = model(img)

output

# %%
# get the class with the highest probability

_, predicted = torch.max(output, 1)

predicted
# %%

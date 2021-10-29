from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

#パラメータの初期化
classes = ["dandelion", "wood sorrel","pansy","trifolium repens","majalis","azalea","viola mandshurica","hyacinth","rhoeas","cherry blossom"]
num_classes = len(classes)
image_size = 150

X = []
Y = []

for index, classlabel in enumerate(classes):
    photo_dir = "data/" + classlabel
    print(photo_dir)
    files = glob.glob(photo_dir + "/*.jpg")
    print(len(files))
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image) /255.0
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

#print(X, Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
print(len(X_train), len(X_test), len(y_train), len(y_test))
np.save("./data/imagefiles.npy",xy)

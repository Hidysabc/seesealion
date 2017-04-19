import numpy as np
import os
import glob
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import skimage.feature

classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups", "error"]

PRJ = "/workspace/seesealion"
DATA_PATH = os.path.join(PRJ,"data","Kaggle-NOAA-SeaLions_FILES")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "Train")
TRAIN_DOTTED_DATA_PATH = os.paht.join(DATA_PATH, "TrainDotted")

alltrainimgs = glob.glob(os.path.join(TRAIN_DATA_PATH, "*.jpg"))
alltraindottedimgs = glob.glob(os.path.join(TRAIN_DOTTED_DATA_PATH, "*.jpg"))



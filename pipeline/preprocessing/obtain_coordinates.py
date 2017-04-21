import numpy as np
import os
import glob
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import skimage.feature
import logging
import multiprocessing as mp

NCPU = mp.cpu_count()
classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups", "error"]
cols = ["id","coordinate_x", "coordinate_y", "sealion_type"]

PRJ = "/workspace/seesealion"
DATA_PATH = os.path.join(PRJ,"data","Kaggle-NOAA-SeaLions_FILES")
OUTPUT_PATH = os.path.join(DATA_PATH, "coordinates")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "Train")
TRAIN_DOTTED_DATA_PATH = os.path.join(DATA_PATH, "TrainDotted")
MISMATCH_INFO_PATH = os.path.join(PRJ,"data","MismatchedTrainimages.txt")
alltrainimgs = glob.glob(os.path.join(TRAIN_DATA_PATH, "*.jpg"))
#alltraindottedimgs = glob.glob(os.path.join(TRAIN_DOTTED_DATA_PATH, "*.jpg"))
id = [os.path.basename(i).partition('.')[0] for i in alltrainimgs if os.path.basename(i).partition('.')[0].isdigit()]
#id = sorted(id,key=lambda item:(int(item),item))


FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOG_MAP = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warn": logging.WARNING
        }

logging.basicConfig(format=FORMAT)
logger = logging.getLogger("obtain_coordinates")
logger.setLevel(LOG_MAP["debug"])

def read_mismatch_info(mismatch_info_path):
    with open(mismatch_info_path) as f:
        lines = f.readlines()
    mismatch_info = [x.strip('\n') for x in lines]

    return mismatch_info

def compare_mismatch_info(image_id, mismatch_info):
    if image_id in mismatch_info:
        logger.debug("Image {} is mismatched!".format(image_id))
        return True
    else:
        return False

#meta_output = []
def obtain_coordinates(i):
    output_file = os.path.join(OUTPUT_PATH, "meta_{}.csv".format(i))
    if os.path.exists(output_file):
        logger.debug("meta_{}.csv already exists!".format(i))
        return
    mismatch_info = read_mismatch_info(MISMATCH_INFO_PATH)
    #logger.debug("Mismatch Info List:", mismatch_info)

    if compare_mismatch_info(i,mismatch_info) is True:
        return
    # read the Train and Train Dotted images
    image_1 = cv2.imread(os.path.join(TRAIN_DOTTED_DATA_PATH, i + ".jpg"))
    image_2 = cv2.imread(os.path.join(TRAIN_DATA_PATH, i + ".jpg"))

    if image_1.shape!=image_2.shape:
        logger.debug("Train Dotted Image {} and Train Image {} mismatch!".format(i,i))
        return
    # absolute difference between Train and Train Dotted images
    try:
        image_3 = cv2.absdiff(image_1, image_2)
    except OSError as err:
        logger.warn("OSError: {}".format(err))
        return
    except:
        logger.warn("Unexpcected error: {}".format(sys.exec_info()[0]))
        return

    #logger.debug("Doing masking...")
    # mask out blackened regions from Train Dotted images
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20]= 0
    mask_1[mask_1 > 0] = 255

    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255

    image_4 = cv2.bitwise_or(image_3, image_3, mask= mask_1)
    image_5 = cv2.bitwise_or(image_4, image_4, mask = mask_2)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = cv2.cvtColor(image_5, cv2.COLOR_BGR2GRAY)

    #logger.debug("Doing blob detection...")
    # blob detection
    blobs = skimage.feature.blob_log(image_6, min_sigma = 3, max_sigma =4, num_sigma = 1, threshold = 0.02)

    # prepare the image to plot results on
    iamge_7 = cv2.cvtColor(image_6, cv2.COLOR_GRAY2BGR)

    #logger.debug("Getting coordinates for blob detection of image_id:{}".format(i))
    meta_output = []
    for blob in blobs:
        # get the coordinates for each blob
        y,x,s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b,g,r = image_1[int(y)][int(x)][:]

        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if r > 200 and b < 50 and g < 50: # RED
            meta_output.append((i,x,y,classes[0]))
        elif r > 200 and b > 200 and g < 50: # MAGENTA
            meta_output.append((i,x,y, classes[1]))
        elif r < 100 and b < 100 and 150 < g < 200: # GREEN
            meta_output.append((i,x,y, classes[4]))
        elif r < 100 and 100 < b and g < 100: # BLUE
            meta_output.append((i,x,y,classes[3]))
        elif r < 150 and b < 50 and g < 100: # BROWN
            meta_output.append((i,x,y,classes[2]))
        else:                               # ERROR
            meta_output.append((i,x,y, classes[5]))

    if len(meta_output) > 0:
        meta_df = pd.DataFrame.from_records(meta_output)
        meta_df.columns = cols
        meta_df.to_csv(output_file, index = False)
        logger.info("Finished output meta_csv for image_id: {}".format(i))

pool = mp.Pool(min(len(id),NCPU))
res = pool.map_async(obtain_coordinates,id)
out = res.get()

logger.info("Done! :)")


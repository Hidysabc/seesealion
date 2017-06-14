import numpy as np
import pandas as pd
import os
import itertools
import cv2
import glob
import logging
import multiprocessing as mp

NCPU = mp.cpu_count()

FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOG_MAP = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warn": logging.WARNING
        }

logging.basicConfig(format=FORMAT)
logger = logging.getLogger('binaryclassification')

logger.setLevel(LOG_MAP["debug"])


PRJ = "/workspace/seesealion"
DATA_PATH = os.path.join(PRJ,"data/Kaggle-NOAA-SeaLions_FILES")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "Train")
TRAIN_DOTTED_DATA_PATH = os.path.join(DATA_PATH,"TrainDotted")
COORDINATES_PATH = os.path.join(DATA_PATH, "coordinates")
TILES_PATH = os.path.join(DATA_PATH,"tiles")
if not os.path.exists(TILES_PATH):
    os.makedirs(TILES_PATH)
    logger.debug("Make new directory since {} doesn't exist.".format(TILES_PATH))

tilesize = 299
images = glob.glob(os.path.join(TRAIN_DATA_PATH,"*.jpg"))
coordinates = glob.glob(os.path.join(COORDINATES_PATH,"*.csv"))
image_ids = [os.path.basename(coordinate).strip(".csv").split("_")[1]
             for coordinate in coordinates]
validate_ratio = 0.15
iid = image_ids

def tiles_binary_classification(iid):
    logger.debug("Doing tiles partitions and binary classification on image_id:{}"\
             .format(iid))
    output_file = os.path.join(TILES_PATH,"img_{}_tile_meta_data.csv".format(iid))
    if os.path.exists(output_file):
        logger.debug("{} already exists!".format(output_file))
        return

    meta = pd.read_csv(os.path.join(COORDINATES_PATH,"meta_{}.csv"\
                        .format(iid)))
    image = cv2.imread(os.path.join(TRAIN_DATA_PATH,"{}.jpg".format(iid)))

    jmax = int(image.shape[0]/tilesize + 0.5)
    imax = int(image.shape[1]/tilesize + 0.5)

    it = itertools.product(range(jmax),range(imax))
    output = []
    for (j,i) in it:
        x_min = i*tilesize
        x_max = (i+1)*tilesize
        y_min = j*tilesize
        y_max = (j+1)*tilesize
        if x_max > image.shape[1]:
            x_max = image.shape[1]
            x_min = x_max - tilesize
        if y_max > image.shape[0]:
            y_max = image.shape[0]
            y_min = y_max - tilesize
        tilename = "{}_{}_{}.npy".format(iid,i,j)
        if os.path.exists(os.path.join(TILES_PATH,tilename)):
            logger.debug("{} already exists!".format(tilename))
            continue
        logger.debug("Processing Tile: {}".format(tilename))

        slt = meta[(meta.coordinate_x > x_min)
                & (meta.coordinate_x < x_max)
                & (meta.coordinate_y > y_min)
                & (meta.coordinate_y < y_max)].sealion_type
        if (len(slt.unique()) == 1 and slt.unique()[0] == 'error'):
            logger.debug("Ignore this tile since only errors labels in the tile.")
            continue
        if (slt.unique().size==0):
            label = 0
        else:
            label = 1

        output.append((iid,tilename,x_min, x_max, y_min, y_max, label))
        tile = image[y_min:y_max,x_min:x_max,:]
        np.save(os.path.join(TILES_PATH,tilename),tile)
        logger.debug("Shape of tile is: {}".format(tile.shape))

    df_output = pd.DataFrame.from_records(output)
    df_output.columns = ["image_id", "tile_filename", "x_min", "x_max", "y_min", "y_max", "class"]
    df_output.to_csv(os.path.join(TILES_PATH,"img_{}_tile_meta_data.csv".format(iid)),index=False)
    logger.debug("Successfully saving img_{}_tile_meta_data.csv!".format(iid))

pool = mp.Pool(min(len(iid),NCPU))
res = pool.map_async(tiles_binary_classification, iid)
out = res.get()

logger.info("Tiles binary classification is done! :)")
logger.info("Concating all the tiles info into one csv_file...")
csv_files = glob.glob(os.path.join(TILES_PATH,"*.csv"))
output_all = [pd.read_csv(single_csv) for single_csv in csv_files]
tiles_info = pd.concat(output_all,axis=0)
tiles_info = tiles_info.set_index('tile_filename')
logger.info("Split images into train and valid sets")

# Generate train and valid image sets
images = np.unique(tiles_info['image_id'])
valid_images = np.random.choice(images, int(len(images)*validate_ratio+0.5))

tiles_info['set'] = 'train'
tiles_info.loc[tiles_info.image_id.isin(valid_images), 'set'] = 'valid'

tiles_info.to_csv(os.path.join(DATA_PATH,'tiles_info_all.csv'))


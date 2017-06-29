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
logger = logging.getLogger('subclassification')

logger.setLevel(LOG_MAP["debug"])


PRJ = "/workspace/seesealion"
DATA_PATH = os.path.join(PRJ,"data/Kaggle-NOAA-SeaLions_FILES")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "Train")
TRAIN_DOTTED_DATA_PATH = os.path.join(DATA_PATH,"TrainDotted")
COORDINATES_PATH = os.path.join(DATA_PATH, "coordinates")
TILES_PATH = os.path.join(DATA_PATH,"tiles_32")
if not os.path.exists(TILES_PATH):
    os.makedirs(TILES_PATH)
    logger.debug("Make new directory since {} doesn't exist.".format(TILES_PATH))

tilesize = 32
images = glob.glob(os.path.join(TRAIN_DATA_PATH,"*.jpg"))
coordinates = glob.glob(os.path.join(COORDINATES_PATH,"*.csv"))
image_ids = [os.path.basename(coordinate).strip(".csv").split("_")[1]
             for coordinate in coordinates]
validate_ratio = 0.15
iid = image_ids

def tiles_subclassification(iid):
    logger.debug("Doing tiles partitions and binary classification on image_id:{}"\
             .format(iid))
    output_file = os.path.join(TILES_PATH,"img_{}_tile_meta_data.csv".format(iid))
    if os.path.exists(output_file):
        logger.debug("{} already exists!".format(output_file))
        return

    meta = pd.read_csv(os.path.join(COORDINATES_PATH,"meta_{}.csv"\
                        .format(iid)))
    meta = meta[meta.sealion_type!='error']
    image = cv2.imread(os.path.join(TRAIN_DATA_PATH,"{}.jpg".format(iid)))

    output = []
    maps = {'adult_males':0,'subadult_males':1, 'adult_females':2, 'juveniles':3,'pups':4,'none':5}
    for _, row in meta.iterrows():
        j = row.coordinate_y
        i = row.coordinate_x
        sealion_type = row.sealion_type
        sealion_class = maps[sealion_type]
        x_min = int(i-tilesize/2)
        x_max = int(i+tilesize/2)
        y_min = int(j-tilesize/2)
        y_max = int(j+tilesize/2)
        if x_max > image.shape[1]:
            x_max = image.shape[1]
            x_min = x_max - tilesize
        if y_max > image.shape[0]:
            y_max = image.shape[0]
            y_min = y_max - tilesize
        tilename = "{}_{}_{}_sealion.npy".format(iid,int(i),int(j))
        if os.path.exists(os.path.join(TILES_PATH,tilename)):
            logger.debug("{} already exists!".format(tilename))
            continue
        logger.debug("Processing Tile: {}".format(tilename))
        '''
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
        '''
        output.append((iid,tilename,x_min, x_max, y_min, y_max, sealion_type, sealion_class))
        tile = image[y_min:y_max,x_min:x_max,:]
        np.save(os.path.join(TILES_PATH,tilename),tile)
        logger.debug("Shape of tile is: {}".format(tile.shape))
    jmax = int(image.shape[0]/tilesize + 0.5)
    imax = int(image.shape[1]/tilesize + 0.5)

    it = itertools.product(range(jmax),range(imax))

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
            sealion_type = 'none'
            sealion_class = maps[sealion_type]
            output.append((iid,tilename,x_min, x_max, y_min, y_max, sealion_type, sealion_class))
            tile = image[y_min:y_max,x_min:x_max,:]
            np.save(os.path.join(TILES_PATH,tilename),tile)
            logger.debug("Shape of tile is: {}".format(tile.shape))
        else:
            logger.debug("This tile contains sealion, and thus we skip over")
            continue


    df_output = pd.DataFrame.from_records(output)
    df_output.columns = ["image_id", "tile_filename", "x_min", "x_max", "y_min", "y_max", "sealiontype","class"]
    df_output.to_csv(os.path.join(TILES_PATH,"img_{}_tile_meta_data.csv".format(iid)),index=False)
    logger.debug("Successfully saving img_{}_tile_meta_data.csv!".format(iid))

pool = mp.Pool(min(len(iid),NCPU))
res = pool.map_async(tiles_subclassification, iid)
out = res.get()

logger.info("Tiles subclassification is done! :)")
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

tiles_info.to_csv(os.path.join(DATA_PATH,'tiles32_info_all.csv'))


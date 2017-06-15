from __future__ import print_function
from __future__ import division
import numpy as np
import os
import logging
import pandas as pd
from preprocessing_augmentation import SeaLionImageDataGenerator as SLIDG
from keras.applications import inception_v3 as I_v3
#from scipy import stats
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, GlobalAveragePooling2D
#from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
#from keras.layers.convolutional import Convolution3D
#from keras.layers.pooling import MaxPooling3D
from keras.optimizers import RMSprop, Adam
import boto3
#from resnet3d import Resnet3DBuilder
import sys
from callbacks import ModelCheckpointS3

model_name = "hw-inceptionv3"

s3bucket = "seesealion-stage1-images"
#input_sample_images = "sample_images"
#input_preprocessing_images = "preprocessing_images"
#input_csv = "csv"
input_dir = '/tmp/seesealion/data/'
batch_size = 32
NB_CLASSES = 2
train_ratio = 0.85
validate_ratio = 0.15
nb_epoch = 100

FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOG_MAP = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warn": logging.WARNING
        }

logging.basicConfig(format=FORMAT)
logger = logging.getLogger(model_name)

logger.setLevel(LOG_MAP["debug"])



s3 = boto3.resource('s3')
bucket = s3.Bucket(s3bucket)
'''
all_keys = [obj.key for obj in bucket.objects.all()]
all_keys = [i for i in all_keys if input_preprocessing_images in i]
patient_id_keys = [filename.split('/') for filename in all_keys]
patient_id_keys = [i[1] for i in patient_id_keys if i[1]!=""]
patient_id = np.unique(patient_id_keys)
patient_id.sort()
patient_ids = [patient.replace('.npy','') for patient in patient_id]

csv_keys = [obj.key for obj in bucket.objects.all()]
csv_keys = [i for i in csv_keys if input_csv in i]
csv_info = [i for i in csv_keys if "sample_img_info" in i]
'''
s3_client = boto3.client('s3')

'''
Don't do download here. Try EBS volume...20170408 Hidy chiu

if not os.path.exists(os.path.join(input_dir, csv_info[0])):
    s3_client.download_file(s3bucket, csv_info[0], os.path.join(input_dir,csv_info[0]))

if not os.path.exists(os.path.join(input_dir, input_preprocessing_images)):
    os.mkdir(os.path.join(input_dir,input_preprocessing_images))
    [s3_client.download_file(s3bucket, os.path.join(input_preprocessing_images,patient), os.path.join(input_dir,input_preprocessing_images,patient)) for patient in patient_id]

'''

'''
labels_info = pd.read_csv('/preprocessing_images_all/sample_images_.csv',
                          index_col=0)
'''
'''
labels_info = pd.read_csv(os.path.join(input_dir, csv_info[0]))
CANCER_MAP = labels_info.set_index('id')['cancer'].to_dict()
labels_info.set_index("id", drop=True, inplace=True)
'''


'''
tmp = [(img.replace('.npy', ''), np.load(os.path.join(path, img))) for img in images]
tmp = [img for img in tmp if img[0] in CANCER_MAP]
X = np.array([x[1] for x in tmp])
X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1))
y = np_utils.to_categorical([CANCER_MAP[c[0]] for c in tmp])
'''

if len(sys.argv)>1:
    logger.info("Start loading model...")
    model_path = os.path.join(input_dir, sys.argv[1])
    s3_client.download_file(s3bucket, sys.argv[1], model_path)
    model = load_model(model_path)
else:
    logger.info("Start building model...")
    base_model = I_v3.InceptionV3(input_shape = (299,299,3),weights='imagenet',classes = 2,include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

train_datagen = SLIDG(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

validate_datagen = SLIDG(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#directory = '/workspace/seesealion/data/Kaggle-NOAA-SeaLions_FILES/tiles'
#csv_path = '/workspace/seesealion/data/Kaggle-NOAA-SeaLions_FILES/tiles_info_all.csv'

directory = '/tmp/seesealion/data/Kaggle-NOAA-SeaLions_FILES/tiles'
csv_path = '/tmp/seesealion/data/Kaggle-NOAA-SeaLions_FILES/tiles_info_all.csv'


tiles_info = pd.read_csv(csv_path).set_index('tile_filename')
'''
#TODO generate train and valid image sets
images = np.unique(tiles_info['image_id'])
valid_images = np.random.choice(images, int(len(images)*validate_ratio+0.5))

tiles_info['set'] = 'train'
tiles_info.loc[tiles_info.image_id.isin(valid_images), 'set'] = 'valid'

tiles_info.to_csv('tiles_info_all.csv')
'''
df_csv_class_train = tiles_info.loc[tiles_info['set'] == 'train', 'class']
df_csv_class_valid = tiles_info.loc[tiles_info['set'] == 'valid', 'class']

train_gen = train_datagen.flow_from_directory_numpy(directory = directory, df_csv_class = df_csv_class_train, batch_size= 2)
validate_gen = train_datagen.flow_from_directory_numpy(directory = directory, df_csv_class = df_csv_class_valid, batch_size= 2)


checkpointer = ModelCheckpointS3(monitor='val_loss',filepath="/tmp/{}-best.hdf5".format(model_name),
                                 bucket = s3bucket,
                                 verbose=0, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss")

logger.info("Start training...")
history = model.fit_generator(train_gen,
                              steps_per_epoch = 10000 ,
                              epochs = nb_epoch,
                              verbose = 1,
                              validation_data= validate_gen,
                              validation_steps = 10000,
                              callbacks= [checkpointer, reduce_lr])

val_loss = history.history["val_loss"]
val_acc = history.history["val_acc"]

best_model_name = "{model_name}-best-{val_acc:.4f}-{val_loss:.4f}.hdf5".format(
    model_name = model_name, val_acc = val_acc[np.argmin(val_loss)], val_loss = np.min(val_loss)
)

copy_source = {
    "Bucket": s3bucket,
    "Key": "{}-best.hdf5".format(model_name)
}

s3_client.copy(copy_source, s3bucket, best_model_name)

logger.info("save final model")

final_model_name = "{model_name}-{epochs}-{val_acc:.4f}-{val_loss:.4f}.hdf5"\
    .format(model_name = model_name, epochs = nb_epoch,
            val_acc = val_acc[-1], val_loss = val_loss[-1])

final_model_path = "/tmp/{}".format(final_model_name)
model.save(final_model_path)
s3_client.upload_file(final_model_path, s3bucket, final_model_name)

#model.save("{}.h5".format(model_name))
logger.info("Finish! :)")
#history = model.fit(X, y, nb_epoch=3, batch_size=2)



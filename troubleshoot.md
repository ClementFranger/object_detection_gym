for tensorflow : python 3.8.6
                 python 64 bit
                 pycharm 2020

buld protoc

CUDA toolkit for gpu augmentation, cdnn download and CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2 as env var
no need to build csv from xml and then csv to tfrecords -> go directly from xml files to tfrecords

todo : what is diff between sample config from tensorflow dir and pipeline config of downloaded model
TODO : tensorboard

add tensorflow models, models/research, slim into PYTHONPATH env var

image/encoded and image/format as feature name or it won't work

fine_tune_checkpoint file is download model ckpt-0 file.
for file name ckpt-0.data-00000-of-00001 -> ckpt-0

ROADMAP : 
# split images to train and test datasets
Images.split()
# generates tfrecords
LabelIMG.write()
# choose model and modify pipeline.config



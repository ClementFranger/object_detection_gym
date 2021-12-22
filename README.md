STEP 1 : Takes images (.jpg) and label them with labelimg
STEP 2 : Split the labels.xml into test and training
STEP 3 : Create label.pbtxt for tensorflow
STEP 4 : Create tfrecords by running test_data.test_tfrecords_write
STEP 5 : Import non trained model into game/models
STEP 6 : Run test_model.test_update to update pipeline_config
STEP 7 : Make environment variable PYTHON to your python exe and TENSORFLOW to the tensorflow folder
STEP 8 : Run test_model.test_train to train your model. Run test_model.test_tensorboard at the same time to monitor
STEP 9 : Run test_model.test_save to save your model
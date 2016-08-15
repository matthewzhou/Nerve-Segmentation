import numpy as np
np.random.seed(2016)
import os, os.path
import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd
import datetime
import time
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam, Adamax
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
import matplotlib as plt


def get_im_cv2(path, img_rows, img_cols):
    #Retrieve the image from the input path and resize it to the user defined parameters
    img = cv2.imread(path, 0)
    resized = cv2.resize(img, (img_cols, img_rows), interpolation = cv2.INTER_LINEAR)
    return resized


def load_train(img_rows, img_cols):
    #Load in training data and masks
    X_train = []
    X_train_id = []
    mask_train = []
    start_time = time.time()

    print('Read train images')
    files = glob.glob("../P5_Submission_Folder/raw/trainsample/*[0-9].tif")
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_train.append(img)
        X_train_id.append(flbase[:-4]) #append unique ids for images 
        mask_path = "../P5_Submission_Folder/raw/trainsample/" + flbase[:-4] + "_mask.tif"
        mask = get_im_cv2(mask_path, img_rows, img_cols)
        mask_train.append(mask)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, mask_train, X_train_id


def load_test(img_rows, img_cols):
    #Load test data
    print('Read test images')
    files = glob.glob("../P5_Submission_Folder/raw/testsample/*[0-9].tif")
    X_test = []
    X_test_id = []
    total = 0
    start_time = time.time()
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_test.append(img)
        X_test_id.append(flbase[:-4]) #unique ids
        total += 1

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id


def rle_encode(img, order='F'):
    #takes in an image and converts numerical ranges to a list of pairs
    #returns the run length encoding of the image 
    #first number is the starting value, second value is the length of the range
    #[1,2,3,5,6,7,9,10,11] becomes [(1,3),(5,3),(9,3)}
    #This format is necessary for the Kaggle submission guidelines but is not relevant to the CNN
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1

    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r)) 
                pos += r
                r = 0
            pos += 1
        else:
            r += 1
    
    if r != 0:
        runs.append((pos, r))
        pos += r
        
    #generate a string of the list pairs
    z = ''
    for rr in runs:
        z += str(rr[0]) + ' ' + str(rr[1]) + ' '
    return z[:-1]


def find_best_mask():
    #adjust file path for raw data directory
    files = glob.glob(os.path.join("/Users/matthewzhou/Desktop/Nerve/P5_Submission_Folder/", "raw", "trainsample", "*_mask.tif"))
    overall_mask = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    overall_mask.fill(0)
    overall_mask = overall_mask.astype(np.float32)

    for fl in files:
        mask = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
        overall_mask += mask
    overall_mask /= 255
    max_value = overall_mask.max()
    koeff = 0.5
    #if the overall_mask pixel value is 
    overall_mask[overall_mask < koeff * max_value] = 0
    overall_mask[overall_mask >= koeff * max_value] = 255
    overall_mask = overall_mask.astype(np.uint8)
    return overall_mask


def create_submission(predictions, test_id, info):
    #function to create submission file for Kaggle
    sub_file = os.path.join('submission_' + info + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    subm = open(sub_file, "w")
    mask = find_best_mask() #
    encode = rle_encode(mask)
    subm.write("img,pixels\n")
    for i in range(len(test_id)):
        subm.write(str(test_id[i]) + ',')
        if predictions[i][1] > 0.5:
            subm.write(encode)
        subm.write('\n')
    subm.close()


def get_empty_mask_state(mask):
    #if sum of pixels for mask file is 0, then set it to empty. Otherwise return 1
    out = []
    for i in range(len(mask)):
        if mask[i].sum() == 0:
            out.append(0)
        else:
            out.append(1)
    return np.array(out)


def read_and_normalize_train_data(img_rows, img_cols):
    train_data, train_target, train_id = load_train(img_rows, img_cols)
    #train target principal components = 200  -- 99.113% of variance explained
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    #reshape data array to format that can be read into the CNN 
    train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    
    train_target = get_empty_mask_state(train_target)
    train_target = np_utils.to_categorical(train_target, 2)
    train_data = train_data.astype('float32')
    train_data /= 255 #normalize train data to 0-1 range
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data(img_rows, img_cols):
    test_data, test_id = load_test(img_rows, img_cols)
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_model(img_rows, img_cols):
    model = Sequential() #initialize model
    model.add(Convolution2D(4, 3, 3, border_mode='same', activation='relu', init='he_normal',
                            input_shape=(1, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    adm = Adamax()
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=adm, loss='categorical_crossentropy')
    return model

def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def getPredScorePercent(train_target, train_id, predictions_valid):
    #play around with how valid predictions are determined
    perc = 0
    for i in range(len(train_target)):
        pred = 1
        if predictions_valid[i][0] > 0.5:
            pred = 0
        real = 1
        if train_target[i][0] > 0.5:
            real = 0
        if real == pred:
            perc += 1
    perc /= len(train_target)
    return perc

def run_cross_validation(nfolds=10):
    img_rows, img_cols = 32, 32 #larger image sizes create more features but risks overfitting
    batch_size = 32 #larger batches will be more computationally expensive and could overfit
    nb_epoch = 50 #trains on more rounds per epoch -- set higher if performing a lot of regularization
    random_state = 51 #set random seed

    train_data, train_target, train_id = read_and_normalize_train_data(img_rows, img_cols)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols)

    yfull_train = dict()
    yfull_test = []
    histlist = []

    #cross-validate across n-folds 
    kf = KFold(len(train_data), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    fold_scores = []
    for train_index, test_index in kf: #generate CV indices for train/valid sets
        model = create_model(img_rows, img_cols)
        X_train, X_valid = train_data[train_index], train_data[test_index]
        Y_train, Y_valid = train_target[train_index], train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        #use callbacks to stop the epoch early if there is sustained stoppage of log loss change
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ] #patience indicates how many rounds before stopping

        #assign log of loglosses for each epoch to list of epoch metrics
        histlist.append(model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks))

        predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        fold_scores.append(score)
        # Store validation predictions to be compared to the final validation set
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        # Store test predictions for this nfold
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction) 


    predictions_valid = get_validation_predictions(train_data, yfull_train)
    score = log_loss(train_target, predictions_valid)
    print("Log_loss train independent avg: ", score)

    print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, img_rows, img_cols, nfolds, nb_epoch))
    perc = getPredScorePercent(train_target, train_id, predictions_valid)
    print('Percent success: {}'.format(perc))

    info_string = 'loss_' + str(score) \
                    + '_r_' + str(img_rows) \
                    + '_c_' + str(img_cols) \
                    + '_folds_' + str(nfolds) \
                    + '_ep_' + str(nb_epoch)
    print "Validation Scores Per Fold: " + str(fold_scores) #print scores for each n-fold
    losslist = pd.DataFrame()
    for i in histlist:
        temp = pd.DataFrame(i.history['loss'])
        losslist = pd.concat([losslist, temp])
    #losslist.to_csv("losslistdropout35.csv") #generate .csv file of losses
    #merge predictions from different n-folds for an averaged aggregate
    test_res = merge_several_folds_mean(yfull_test, nfolds)

    #creates submission file of run length encoded mask pixels to Kaggle
    #create_submission(test_res, test_id, info_string)

    
if __name__ == '__main__':
    run_cross_validation(10)


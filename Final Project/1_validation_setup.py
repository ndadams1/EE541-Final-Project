import os
import cv2
import numpy as np

train_dir = "/home/nick/Documents/Datasets/ASL/asl_alphabet_train"
test_dir = "/home/nick/Documents/Datasets/ASL/test_set"

def get_data(data_dir) :
    imtrain = []
    imtest = []
    labtrain = []
    labtest = []
    
    dir_list = os.listdir(data_dir)
    dir_list.sort()
    label_eq = dir_list
    for i in range(len(dir_list)):
        print("Obtaining images of", dir_list[i], "...")
        count = 0
        for image in os.listdir(data_dir + "/" + dir_list[i]):
            img = cv2.imread(data_dir + '/' + dir_list[i] + '/' + image)
            img = cv2.resize(img, (125, 125))
            if count > 2699 or count <= 299:
                imtest.append(img)
                labtest.append(i)
            else:
                imtrain.append(img)
                labtrain.append(i)
            count += 1
    
    return imtrain, imtest, labtrain, labtest, label_eq
        
imtrain, imtest, labtrain, labtest, label_eq = get_data(train_dir)

for i in range(len(imtrain)):
    imtrain[i] = imtrain[i].T
x_train = np.array(imtrain)
y_train = np.array(labtrain)

for i in range(len(imtest)):
    imtest[i] = imtest[i].T
x_test = np.array(imtest)
y_test = np.array(labtest)

import h5py
file = '1_validimg.hdf5'
with h5py.File(file, 'w') as hf:
    hf.create_dataset('x_train', data = np.asarray(x_train))
    hf.create_dataset('x_test', data = np.asarray(x_test))
    hf.create_dataset('y_train', data = np.asarray(y_train))
    hf.create_dataset('y_test', data = np.asarray(y_test))
print(label_eq)
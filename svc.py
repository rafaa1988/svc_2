import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

root_dir = 'new_training/'
norms_dir = 'one/'
deffs_dir = 'two/'

all_norms = os.listdir(os.path.join(root_dir, norms_dir))
all_deffs = os.listdir(os.path.join(root_dir, deffs_dir))


import cv2 as img
from skimage.transform import resize

target_w = 70
target_h = 70
all_imgs_orig = []
all_imgs = []
all_labels = []
idx = 0 


for img_name in all_norms:
    img_arr = img.imread(os.path.join(root_dir, norms_dir, img_name))
    w,h,d = img_arr.shape
    img_arr_rs = img_arr
    img_arr_rs = resize(img_arr, (target_w, target_h), mode='reflect')
    all_imgs.append(img_arr_rs)
    all_imgs_orig.append(img_arr)
    all_labels.append(1)

for img_name in all_deffs:
    img_arr = img.imread(os.path.join(root_dir, deffs_dir, img_name))
    w,h,d = img_arr.shape
    img_arr_rs = img_arr
    img_arr_rs = resize(img_arr, (target_w, target_h), mode='reflect')
    all_imgs.append(img_arr_rs)
    all_imgs_orig.append(img_arr)
    all_labels.append(0)
    

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X = np.array(all_imgs)
Y = to_categorical(np.array(all_labels),num_classes=2)
Y = Y[:,0]

n,w,l,d = X.shape
X_raw = np.reshape(X,(n,w*l*d))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)


from sklearn import svm

C_values = [0.00075,0.001,0.005,0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0,100.0,500.0]
C_val = 1
prediction_acc = []
train_acc = []
models = (svm.SVC(kernel='linear', C=C_val),
          svm.SVC(kernel='rbf', C=C_val),
          svm.SVC(kernel='rbf', gamma=10, C=C_val),
          svm.SVC(kernel='sigmoid', C=1, gamma=0.1, coef0=0.25),
          svm.SVC(kernel='poly', degree=3, gamma=0.5, coef0=0.25, C=C_val))
    
plt.figure(1, figsize=(13, 10))
plt.clf()

for idx,model in enumerate(models):
    for C_val in C_values:
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2)
        model.fit(X_train,Y_train)
        tr_acc = model.score(X_train,Y_train)
        ts_acc = model.score(X_test,Y_test)
        train_acc.append(tr_acc)
        prediction_acc.append(ts_acc)
        print("Model:%d, Train Accuracy:%f, Test Accuracy:%f, C Value:%f"%(idx,tr_acc,ts_acc,C_val))
    print('-------')
    aa = model.get_params()
    if 'linear' in aa.values():
#       plt.plot(range(len(C_values)),train_acc[:13],color='k',label='train-linear')
        plt.plot(range(len(C_values)),prediction_acc[:13],color='r',label='test-linear')
        
    elif 'rbf' in aa.values():
#       plt.plot(range(len(C_values)),train_acc[13:26],color='k',label='train-rbf')
        plt.plot(range(len(C_values)),prediction_acc[13:26],color='b',label='test-rbf')
        
    elif '10' in aa.values():
#       plt.plot(range(len(C_values)),train_acc[26:39],color='k',label='train-rbfgamma')
        plt.plot(range(len(C_values)),prediction_acc[26:39],color='c',label='test-rbfgamma')
    elif 'sigmoid' in aa.values():
#       plt.plot(range(len(C_values)),train_acc[39:52],color='k',label='train-sigmoid')
        plt.plot(range(len(C_values)),prediction_acc[39:52],color='g',label='test4-sigmoid')
    else:
#       plt.plot(range(len(C_values)),train_acc[52:],color='k',label='train-poly')
        plt.plot(range(len(C_values)),prediction_acc[52:],color='m',label='test-poly')

plt.legend(loc=1)
plt.show()



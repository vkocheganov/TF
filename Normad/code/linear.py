# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import tensorflow as tf
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print("hello\n")

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
from sklearn.model_selection import train_test_split

t1 = 'formation_energy_ev_natom'
t2 = 'bandgap_energy_ev'
    
def Linear_train(X_t, X_v, y1_t, y1_v, y2_t, y2_v, max_depth=3, n_trees=8, n_steps=1000, learn_rate=.1):
    np.random.seed(1)
    print X_t.corr()

    clf=LinearRegression()
    y_t=np.concatenate((y1_t,y2_t), axis=1)
    y_v=np.concatenate((y1_v,y2_v), axis=1)
    clf.fit(X_t,np.array(y_t))

    if y1_v.shape[0] != 0:
        y_out=clf.predict(X_v)
        y1_out=y_out[:,0]
        y2_out=y_out[:,1]
        print clf.score(X_v,y_v)
        loss1=np.sqrt(np.mean(np.square(np.log(y1_out + 1) - np.log(y1_v+1))))
        loss2=np.sqrt(np.mean(np.square(np.log(y2_out + 1) - np.log(y2_v+1))))
        return np.mean([loss1,loss2])
    else:
        print("create submission")
        sample = pd.read_csv('../input/sample_submission.csv')
        sample.head()
        pred_y1 = clf1.predict(X_v)
        pred_y2 = clf2.predict(X_v)
        subm = pd.DataFrame()
        subm['id'] = sample['id']
        subm['formation_energy_ev_natom'] = pred_y1
        subm['bandgap_energy_ev'] = pred_y2
        subm.to_csv("subm_gbt_%s_%s_%s_%.5f.csv" % (max_depth,n_trees,n_steps,learn_rate), index=False)
        

def cross_validation(train, validation_part, cv_folds, learn_rate):
    # cv_folds=3
    my_loss=[]
    for i in range(cv_folds):
        X_train, X_validation = train_test_split(train, test_size=validation_part)

        y1_train = X_train[t1][:,np.newaxis]
        y2_train = X_train[t2][:,np.newaxis]
        X_train = X_train.drop([t1, t2], axis=1)

        y1_validation = X_validation[t1][:, np.newaxis]
        y2_validation = X_validation[t2][:, np.newaxis]
        X_validation = X_validation.drop([t1, t2], axis=1)

        # print(X_train.shape, y1_train.shape, y2_train.shape)
        # print(X_validation.shape, y1_validation.shape, y2_validation.shape)

        loss_tmp = Linear_train(X_train, X_validation, y1_train, y1_validation, y2_train, y2_validation, learn_rate=learn_rate)
        my_loss.append(loss_tmp)
    m=np.mean(my_loss)
    s=np.std(my_loss)
    my_loss.append(m)
    my_loss.append(s)
    # mystr="%s" %[n_units,n_layers,n_steps,learn_rate]
    # my_loss.append(mystr)
    return my_loss
    
    


train=train.drop(['id'],axis=1)

cv_folds=3
all_errs=[]
# for n_units in range(10,20,1):
#     for n_layers in range(5,10,1):
#for learn_rate in np.arange(0.001,0.01,0.001):
# for learn_rate in np.arange(0.005,0.01,0.001):
for learn_rate in np.arange(0.005,0.006,0.001):
    # n_units=16
    # n_layers=8
    # learn_rate=0.004
    # for n_steps in range(400,600,200):
    my_loss=cross_validation(train, 0.3,cv_folds,learn_rate=0.001)
    all_errs.append(my_loss)
    #print('n_units=%s, n_layers=%s,n_steps=%s, learn_rate=%s, my_loss:%s' %(n_units,n_layers,n_steps,learn_rate,my_loss))

    sys.stdout.flush()

np.set_printoptions(threshold=np.nan)
print("errors:", all_errs)
all_errs=np.array(all_errs)
sorted=all_errs[np.argsort(all_errs[:,cv_folds])]

print("sorted errors", sorted)

# n_units=16
# n_layers=8
# n_steps=400
# learn_rate=0.007

# y1_train = train[t1][:,np.newaxis]
# y2_train = train[t2][:,np.newaxis]
# X_train = train.drop([t1, t2], axis=1)
# X_test = test.drop(['id'], axis=1)

# print(X_train.shape, y1_train.shape, y2_train.shape)

# NN_train(X_train, X_test, y1_train, np.array([]), y2_train, np.array([]),  n_units=n_units, n_layers=n_layers, n_steps=n_steps, learn_rate=learn_rate)



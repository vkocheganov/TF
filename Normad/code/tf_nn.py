# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print("hello\n")

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
from sklearn.model_selection import train_test_split

t1 = 'formation_energy_ev_natom'
t2 = 'bandgap_energy_ev'

    
def NN_train(X_t, X_v, y1_t, y1_v, y2_t, y2_v, n_units=16, n_layers=8, n_steps=500, learn_rate=0.004):
    
    
    tf.set_random_seed(1)
    np.random.seed(1)

    tf.reset_default_graph()


    activation = tf.tanh
    #activation = tf.nn.relu
    #activation = tf.nn.sigmoid

    tf_is_training = tf.placeholder(tf.bool, None)

    tf_x = tf.placeholder(tf.float32, (None, X_t.shape[1]), name='tf_x')
    tf_y1 = tf.placeholder(tf.float32, (None, 1), name='tf_y1')
    tf_y2 = tf.placeholder(tf.float32, (None, 1), name='tf_y2')

    tf_xn = tf.layers.batch_normalization(tf_x, training=tf_is_training, name='tf_xn')

    def add_norm_layer(inputs, nunits, activation, training=False, name=None):
        l = tf.layers.dense(tf_xn, n_units, activation=activation, name=name)
        ln = tf.layers.batch_normalization(l, training=training, name=name + 'n')
        return ln



    # Formation energy
    l = tf_xn
    for i in range(n_layers):
        l = add_norm_layer(l, n_units, activation, training=tf_is_training, name='l%s_y1' % (i + 1))
    output_y1 = tf.layers.dense(l, 1, activation=tf.exp, name='output_y1')
    # Bandgap energy
    l = tf_xn
    for i in range(n_layers):
        l = add_norm_layer(l, n_units, activation, training=tf_is_training, name='l%s_y2' % (i + 1))
    output_y2 = tf.layers.dense(l, 1, activation=tf.exp, name='output_y2')

    # loss
    loss_y1 = tf.sqrt(tf.losses.mean_squared_error(tf.log1p(tf_y1), tf.log1p(output_y1)))
    loss_y2 = tf.sqrt(tf.losses.mean_squared_error(tf.log1p(tf_y2), tf.log1p(output_y2)))
    loss = (loss_y1 + loss_y2) / 2.0

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.05, momentum=1.0)
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    train_op = optimizer.minimize(loss)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_data = []
    for step in range(n_steps):
        # training loss
        _, lt = sess.run([train_op, loss], {tf_x: X_t, tf_y1: y1_t, tf_y2: y2_t, tf_is_training: True})
        
        # validation loss
        if y1_v.shape[0] != 0:
            lv = sess.run(loss, {tf_x: X_v, tf_y1: y1_v, tf_y2: y2_v, tf_is_training: True})
            loss_data.append([step, lt, lv])
        else:
            loss_data.append([step, lt])

    # loss_data = np.array(loss_data)
    #    print(loss_data[-1][1:])
    if y1_v.shape[0] != 0:
        loss, pred_y1, pred_y2 = sess.run([loss, output_y1, output_y2], {tf_x: X_v, tf_y1: y1_v, tf_y2: y2_v, tf_is_training: True})
        return loss
    else:
        print("create submission")
        sample = pd.read_csv('../input/sample_submission.csv')
        sample.head()
        pred_y1, pred_y2 = sess.run([output_y1, output_y2], {tf_x: X_v, tf_is_training: True})
        subm = pd.DataFrame()
        subm['id'] = sample['id']
        subm['formation_energy_ev_natom'] = pred_y1
        subm['bandgap_energy_ev'] = pred_y2
        subm.to_csv("subm_nn_%s_%s_%s_%.5f.csv" % (n_units,n_layers,n_steps,learn_rate), index=False)
        

def cross_validation(train, validation_part, cv_folds, n_units, n_layers, n_steps, learn_rate):
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

        loss_tmp = NN_train(X_train, X_validation, y1_train, y1_validation, y2_train, y2_validation,  n_units=n_units, n_layers=n_layers, n_steps=n_steps, learn_rate=learn_rate)
        my_loss.append(loss_tmp)
    m=np.mean(my_loss)
    s=np.std(my_loss)
    my_loss.append(m)
    my_loss.append(s)
    mystr="%s" %[n_units,n_layers,n_steps,learn_rate]
    my_loss.append(mystr)
    return my_loss
    
    


train=train.drop(['id'],axis=1)

cv_folds=3
all_errs=[]
# for n_units in range(10,20,1):
#     for n_layers in range(5,10,1):
#for learn_rate in np.arange(0.001,0.01,0.001):
# for learn_rate in np.arange(0.005,0.01,0.001):
for learn_rate in np.arange(0.005,0.006,0.001):
    n_units=16
    n_layers=8
# learn_rate=0.004
    for n_steps in range(400,600,200):
        my_loss=cross_validation(train, 0.3,cv_folds, n_units, n_layers, n_steps, learn_rate)
        all_errs.append(my_loss)
#        print('n_units=%s, n_layers=%s,n_steps=%s, learn_rate=%s, my_loss:%s' %(n_units,n_layers,n_steps,learn_rate,my_loss))

        sys.stdout.flush()

np.set_printoptions(threshold=np.nan)
print("errors:", all_errs)
all_errs=np.array(all_errs)
sorted=all_errs[np.argsort(all_errs[:,cv_folds])]

print("sorted errors", sorted)

n_units=16
n_layers=8
n_steps=400
learn_rate=0.007

y1_train = train[t1][:,np.newaxis]
y2_train = train[t2][:,np.newaxis]
X_train = train.drop([t1, t2], axis=1)
X_test = test.drop(['id'], axis=1)

print(X_train.shape, y1_train.shape, y2_train.shape)

NN_train(X_train, X_test, y1_train, np.array([]), y2_train, np.array([]),  n_units=n_units, n_layers=n_layers, n_steps=n_steps, learn_rate=learn_rate)



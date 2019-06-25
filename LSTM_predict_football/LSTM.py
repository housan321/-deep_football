#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.contrib import rnn


# In[2]:


import pandas as pd
graph = tf.Graph()

# In[3]:


data=pd.read_csv("samples/allAtt_onehot_large_train.csv")
dataT=pd.read_csv("samples/allAtt_onehot_large_test.csv")
print(data.head())
print(data.shape)


# In[4]:



hm_epochs=10
n_classes = 2
batch_size = 1
chunk_size=34
n_chunks=1
rnn_size=64

with graph.as_default():
    x = tf.placeholder('float', [None, n_chunks,chunk_size])
    y = tf.placeholder('float')


# In[5]:


def recurrent_neural_model(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    x=tf.transpose(x,[1,0,2])
    print("transpose",x)
    x=tf.reshape(x,[-1,chunk_size])
    print("reshape",x)
    x=tf.split(x,n_chunks)
    print("split",x)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    


    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


# In[6]:


def train_neural_network(x):
    prediction = recurrent_neural_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    y_pred = tf.nn.softmax(logits=prediction)

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        hm_epochs=30
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(0,data.shape[0],batch_size):
                epoch_x, epoch_y = data.iloc[i:i+batch_size,1:35].values,data.iloc[i:i+batch_size,35:].values
                epoch_x=epoch_x.reshape((batch_size,n_chunks,chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy Train:',accuracy.eval({x:data.iloc[:,1:35].values.reshape((-1,n_chunks,chunk_size)), 
                                               y:data.iloc[:,35:].values}))
        print('Accuracy Test:',accuracy.eval({x:dataT.iloc[:,1:35].values.reshape((-1,n_chunks,chunk_size)), 
                                              y:dataT.iloc[:,35:].values}))
        saver.save(sess, "./model/model.ckpt")


# In[7]:

with graph.as_default():
    train_neural_network(x)


# In[ ]:






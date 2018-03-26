# import numpy as np
from utils import *
import os
#
# file_names = os.listdir(args.gaze_path)
#
# mat = input_generator(file_names, 500, 25, 'HR', 50)
#
# print mat
# # count = 0
# # for i in range(0, 900, 2 * ((500 - 50)/ 25)):
# #     count += 1
# #
# #
# # print count
import numpy as np
import tensorflow as tf

# b = tf.get_variable(name='D', shape=[100, 100], initializer=tf.zeros_initializer())
# a = tf.Variable(name='a', initial_value=np.zeros(shape=[100, 100]), trainable=False, dtype=tf.float64)
# n = tf.Variable(name='n', initial_value=np.random.normal(0, 0.01, [100, 3]), trainable=True, dtype=tf.float64)
# file_names = os.listdir(args.gaze_path)
# mat = input_generator(file_names, 500, 25, 'HR', 20)
a = np.zeros(shape=[35, 40])
n = np.random.normal(0, 0.01, [35, 5])
for i in range(35):
    a[i, i:i+5] = n[i]
c = np.transpose(a)
x = np.tile(c, 25)
y = np.tile(x, 102)
b = tf.get_variable('b', initializer=np.tile(np.transpose(a), 102))
b = tf.where(b == 0, tf.stop_gradient(b), b)
shape = tf.shape(b)

# for i in range(98):
#     a = a[i, i:i+3].assign(n[i])
# a = np.transpose(a)
# a1 = a[np.where(a == 0)]
# a2 = tf.get_variable('a2', initializer=tf.constant(a[np.where(a != 0)]))
# a1_stop = tf.stop_gradient(tf.identity(tf.get_variable('a1', initializer=tf.constant(a1))))
# a = tf.concat((a2, a1_stop), axis=0)


# a = tf.constant([[1, -1], [-2, -3]])
#
# a = a[1].assign(tf.get_variable())
# A = tf.Variable([[1., 0.], [3., 0.]])
# A1 = A[:,0:1] # just some slicing of your variable
# A2 = A[:,1:2]
# A2_stop = tf.stop_gradient(tf.identity(A2))
# A = tf.concat((A1, A2_stop), axis=1)
#
model = tf.global_variables_initializer()

with tf.Session() as sess:
    output = sess.run(model)
    o = sess.run(b)
    s = sess.run(shape)

print()
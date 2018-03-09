from Sparse import *


save_path = os.path.join(args.save_dir + 'model_NLR=0.001')
file_names = os.listdir(args.gaze_path)
y_, maxi, mini = normalize_input(input_generator(file_names, 500, 25, 'HR', 50), 'Normalization')
d = tf.get_variable('D', shape=[np.shape(y_)[0], args.dimension2])
x = tf.get_variable('X', shape=[args.dimension2, np.shape(y_)[1]])
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_path)
    print(' Model Restore Successful!')
    d = de_normalized(d.eval(), maxi, mini)
    print('D: ', d)
    x = de_normalized(x.eval(), maxi, mini)
    print(np.max(x), np.min(x))
    print(np.max(d), np.min(d))
    print('X: ', x)

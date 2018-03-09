import os
from utils import *


def main():
    file_names = os.listdir(args.gaze_path)
    save_path = os.path.join(args.save_dir, 'model_')
    y_, maxi, mini = normalize_input(input_generator(file_names, 500, 25, 'HR', 50), 'Normalization')

    with tf.name_scope('Input'):
        y = tf.placeholder(shape=np.shape(y_), name='Y', dtype=tf.float32)

    with tf.name_scope('Weights'):
        d = variable_creator('D', [np.shape(y)[0], args.dimension2])
        tf.summary.histogram('D', d)
        x = variable_creator('X', [args.dimension2, np.shape(y)[1]])
        tf.summary.histogram('X', x)

    with tf.name_scope('Loss'):
        sparse_loss = loss(y, x, d, args.sparsity_coeff, 'Sparse_Loss')
        tf.summary.scalar('Loss', sparse_loss)

    with tf.name_scope('Adam_Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, name='Adam_Opt').minimize(sparse_loss)

    merged = tf.summary.merge_all()
    model = tf.global_variables_initializer()
    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(model)
        summary = tf.summary.FileWriter(logdir="./logs/", graph=sess.graph)
        for iter in range(args.iterations):
            _, summary_tr, loss_r = sess.run([optimizer, merged, sparse_loss], feed_dict={y:y_})
            print("Loss after iteration {0} is: {1}".format(iter, loss_r))
            summary.add_summary(summary_tr)
        saver.save(sess=sess, save_path=save_path + 'NLR=' + str(args.learning_rate))
        summary.close()
        print("Training Complete")
        print(" Sparse X: ", de_normalized(sess.run(x), maxi, mini))
        print(" D: ", de_normalized(sess.run(d), maxi, mini))


if __name__=='__main__':
    main()
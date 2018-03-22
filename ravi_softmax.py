import tensorflow as tf
import tensorflowvisu
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# -------------------- Generate model here -------------------------- #

X = tf.placeholder([None, 28, 28,1])
W = tf.Variable(tf.zeros[784,10])
b = tf.Variable(tf.zeros[10])

init = tf.initialize_all_variables()

# reshape X
X_reshape = tf.reshape(X,[-1,784])

# predictions
Y = tf.nn.softmax(tf.matmul(X_reshape,W) + b)

# one hot encoded correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

# error/loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# check if answer is correct
is_correct = tf.equal(tf.argmax(Y_), tf.argmax(Y))

# compute accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Specify optimizer type
optimizer = tf.train.GradientDescentOptimizer(0.003)

# Specify what function to minimize
train_step = optimizer.minimize(cross_entropy)

# ---------------------  Import data --------------------------------------------- #

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# ---------------------  Now start running the session --------------------------- #

sess = tf.Session()
sess.run(init)

for i in range(1000):

    # load batch of images
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y: batch_Y}

    # train
    sess.run(train_step, feed_dict=train_data)

    # success ?
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    # success on test data ?
    test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    a, c = sess.run([accuracy, cross_entropy], feed=test_data)

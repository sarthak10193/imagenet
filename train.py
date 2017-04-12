import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from vggModel import vgg16

IMAGE_SIZE = 227
num_classes = 3

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_file", "/home/sarthak/PycharmProjects/imagenet/imagenet/trainData/train.txt", "training data path")
flags.DEFINE_string("valid_file", "/home/sarthak/PycharmProjects/imagenet/imagenet/trainData/valid.txt", "validation data path")
flags.DEFINE_float("learning_rate", 0.01, "learning rate")
flags.DEFINE_integer("batch_size",64, "feed dict batch size")
flags.DEFINE_integer("num_epochs", 5, "no of epochs")
flags.DEFINE_float("dropout_prob", 0.50, "drop out prob")
flags.DEFINE_string("filewriter_path", "/home/sarthak/PycharmProjects/imagenet/imagenet/summaries", "summaries path ")
flags.DEFINE_string("checkpoint_path", "/home/sarthak/PycharmProjects/imagenet/imagenet/checkpoint", "checkpoints path")



train_layers = ['fc8', 'fc7']
#train_layers = ['fc8_W', 'fc8_b'] #'fc6_W', 'fc6_b', 'fc7_W', 'fc7_b',


if not os.path.isdir(FLAGS.checkpoint_path):
    os.mkdir(FLAGS.checkpoint_path)


# TF placeholder for graph inputs and labels
images_placeholder_tf = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
labels_placeholder_tf = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)


model = AlexNet(images_placeholder_tf, keep_prob, num_classes, train_layers)
#model = vgg16(images_placeholder_tf, keep_prob, num_classes, train_layers)

# get logits
#logits = model.fc16
logits = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]


with tf.name_scope("cross_ent"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels_placeholder_tf))

# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))
  
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)


for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)


for var in var_list:
  tf.summary.histogram(var.name, var)
  

tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placeholder_tf, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  

tf.summary.scalar('accuracy', accuracy)


merged_summary = tf.summary.merge_all()


writer = tf.summary.FileWriter(FLAGS.filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

train_generator = ImageDataGenerator(FLAGS.train_file,scale_size=[IMAGE_SIZE,IMAGE_SIZE],horizontal_flip = True, shuffle = True, )
val_generator = ImageDataGenerator(FLAGS.valid_file, scale_size=[IMAGE_SIZE,IMAGE_SIZE], shuffle = False, )

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / FLAGS.batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / FLAGS.batch_size).astype(np.int16)



with tf.Session() as sess:
 

  sess.run(tf.global_variables_initializer())
  
  writer.add_graph(sess.graph)
  
  # Load the pre-trained weights of the model
  model.load_initial_weights(sess)

  print("Running Training ......\n")
  for epoch in range(FLAGS.num_epochs):
    
        print("epoch no:", epoch)
        
        step = 1
        
        while step < train_batches_per_epoch:

            batch_xs, batch_ys = train_generator.next_batch(FLAGS.batch_size)
            step+=1


            sess.run(train_op, feed_dict={images_placeholder_tf: batch_xs,
                                          labels_placeholder_tf: batch_ys,
                                          keep_prob: FLAGS.dropout_prob})

            # fetching the summary and loss

            summary, l= sess.run([merged_summary, loss], feed_dict={images_placeholder_tf: batch_xs,
                                                    labels_placeholder_tf: batch_ys,
                                                    keep_prob: 1.})

            print("loss @ step :", step, "  --  ", l)
            writer.add_summary(summary, epoch*train_batches_per_epoch + step)
                
            step += 1
            
        # Validate the model on the entire validation set
        print("Running Validations data eval")
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(FLAGS.batch_size)
            acc = sess.run(accuracy, feed_dict={images_placeholder_tf: batch_tx,
                                                labels_placeholder_tf: batch_ty,
                                                keep_prob: 1.0})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("validation accuracy :", test_acc*100, " %")
        
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now()))  
        
        #save checkpoint of the model
        checkpoint_name = os.path.join(FLAGS.checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


        

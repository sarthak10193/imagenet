import tensorflow as tf
import numpy as np


class vgg16:
    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):

        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH =  '/home/sarthak/PycharmProjects/imagenet/imagenet/preTrainedWeights/vgg16_weights.npz'
        else:
            self.WEIGHTS_PATH = weights_path

        # creating the model graph
        self.create()

    def create(self):
        """
        conv_kernel, kernel_height, kernel_width , number of filter , stride_x, stride_y, padding, scope name

        :return: logits
        """
        print("==== Using VGG16 for imagenet classification ====\n")

        conv1 = conv(self.X, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1_W')

        conv2 = conv(conv1, 3, 3, 64, 1, 1, padding="SAME", name="conv1_2_W")

        pool1 = max_pool(conv2, 2, 2, 2, 2, padding='SAME', name='pool1')

        conv3 = conv(pool1, 3, 3, 128, 1, 1, padding="SAME", name="conv2_1_W")

        conv4 =  conv(conv3, 3, 3, 128, 1, 1, padding="SAME", name="conv2_2_W")

        pool2 = max_pool(conv4, 2, 2, 2, 2, padding='SAME', name='pool2')

        conv5 = conv(pool2, 3, 3, 256, 1, 1, padding="SAME", name="conv3_1_W")
        conv6 = conv(conv5, 3, 3, 256, 1, 1, padding="SAME", name="conv3_2_W")
        conv7 = conv(conv6, 3, 3, 256, 1, 1, padding="SAME", name="conv3_3_W")

        pool3 = max_pool(conv7, 2, 2, 2, 2, padding='SAME', name='pool3')

        conv8 = conv(pool3, 3, 3, 512, 1, 1, padding="SAME", name="conv4_1_W")
        conv9 = conv(conv8, 3, 3, 512, 1, 1, padding="SAME", name="conv4_2_W")
        conv10 = conv(conv9, 3, 3, 512, 1, 1, padding="SAME", name="conv4_3_W")

        pool4 = max_pool(conv10, 2, 2, 2, 2, padding='SAME', name='pool4')

        conv11 = conv(pool4, 3, 3, 512, 1, 1, padding="SAME", name="conv5_1_W")
        conv12 = conv(conv11, 3, 3, 512, 1, 1, padding="SAME", name="conv5_2_W")
        conv13 = conv(conv12, 3, 3, 512, 1, 1, padding="SAME", name="conv5_3_W")

        pool5 = max_pool(conv13, 2, 2, 2, 2, padding='SAME', name='pool5')

        shape = int(np.prod(pool5.get_shape()[1:]))

        flattened = tf.reshape(pool5, [-1, shape])
        fc14 = fc(flattened, shape, 4096, name='fc6_W')
        dropout14 = dropout(fc14, self.KEEP_PROB)

        fc15 = fc(dropout14, 4096, 4096, name='fc7_W')
        dropout15 = dropout(fc15, self.KEEP_PROB)


        self.fc16 = fc(dropout15, 4096, self.NUM_CLASSES, relu=False, name='fc8_W')


    def load_initial_weights(self, session):
            """
            This method loads the pre-trained tensorflow weights from vgg16_weights.npz
            using the scope name of each operation the appropriate non-trainable variables are assigned there values
            The .npz file is of the form <key, value> where key gives the layer-info using the scope_name/variable_name
            and value the appropriate weights

            :param session: current session of tf
            :return:
            """
            weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes')

            keys = sorted(weights_dict.keys())

            for op_name in keys:

                if op_name not in self.SKIP_LAYER:
                    if(op_name[-1:]=='W'):
                        with tf.variable_scope(op_name, reuse=True):
                            #print("modify weights for only:", op_name, weights_dict[op_name].shape)

                            weightsdata  = weights_dict[op_name]
                            biasesdata = weights_dict[op_name.replace("W", 'b')]


                            var1 = tf.get_variable('biases', trainable=False)

                            session.run(var1.assign(biasesdata))


                            var2 = tf.get_variable('weights', trainable=False)
                            session.run(var2.assign(weightsdata))


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME'):
    """

    :param x: convolution kernel
    :param filter_height:
    :param filter_width:
    :param num_filters:
    :param stride_y:
    :param stride_x:
    :param name: scope name of the layer weights and biases
    :param padding:
    :return: relu
    """

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # create or get variables weights and biases under the given name scope.
    with tf.variable_scope(name) as scope:

        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width, input_channels , num_filters])

        biases = tf.get_variable('biases', shape=[num_filters])


        out = tf.nn.conv2d(x, weights,strides=[1, stride_y, stride_x, 1], padding=padding)


        bias = tf.reshape(tf.nn.bias_add(out, biases), out.get_shape().as_list())


        relu = tf.nn.relu(bias, name=scope.name)

        return relu

def fc(x, num_in, num_out, name, relu=True):

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)

        biases = tf.get_variable('biases', [num_out], trainable=True)


        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)





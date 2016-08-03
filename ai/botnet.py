import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import preprocessing
from string import punctuation

class DataFeed(object):
    def __init__(self, data, labels, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=test_size)
        print(len(self.y_train), len(self.y_test))
       
    def get_test_data(self):
        return np.asarray(self.x_test), np.asarray(self.y_test)

    def get_train_data(self):
        return np.asarray(self.x_train), np.asarray(self.y_train)

    def get_next_batch(self, batch_size, phase='train'):
        if phase not in ['train', 'test']:
            sys.exit("`phase` must be in ['train', 'test']")
        batch_indexes = np.random.choice(range(len(getattr(self, 'x_' + phase))), batch_size, replace=False)
        x_batch = np.asarray([getattr(self, 'x_' + phase)[i] for i in batch_indexes])
        y_batch = np.asarray([getattr(self, 'y_' + phase)[i] for i in batch_indexes])
        return x_batch, y_batch


class DataRepresentation(object):
    def __init__(self, filename):
        self.data, self.labels = self.parse(filename)
        self.labels = self.labels_to_one_hot()
        self.sentence_wordwise = [x.split() for x in self.data]
        self.sentence_wordwise = [[x for x in y if x != 'xxx'] for y in self.sentence_wordwise] 
        self.max_words = max([len(x) for x in self.sentence_wordwise])
        self.vocabulary = list(set([x for y in self.sentence_wordwise for x in y]))
        self.size_voc = len(self.vocabulary)
        self.one_hot_mapping = {self.vocabulary[i_]: i_ for i_ in xrange(self.size_voc)}
        self.x = [self.representation(x) for x in self.sentence_wordwise]

    def labels_to_one_hot(self):
        self.list_labels = list(set(self.labels))
        self.label_map = {}
        for label in self.list_labels:
            aux = np.zeros(len(self.list_labels))
            aux[self.list_labels.index(label)] = 1
            self.label_map[label] = aux
        return [self.label_map[x] for x in self.labels] 

    def parse(self, filename):
        with open(filename) as f:
            data_ = [line.rstrip(' \n') for line in f]
        y = [x[x.rfind('/'):] for x in data_]
        y = [x.rstrip(' ') for x in y]
        data = [preprocessing.clean_string(x[:x.rfind('/')].rstrip(' ' + punctuation)) for x in data_]
        return data, y

    def padding(self, representation):
        if len(representation) < self.max_words:
            for i in xrange(self.max_words - len(representation)):
                representation.append(np.zeros(self.size_voc))
        return representation

    def one_hot_encoding(self, sentence):
        representation = []
        for word in sentence:
            representation_ = np.zeros(self.size_voc)
            representation_[self.one_hot_mapping[word]] = 1
            representation.append(representation_)
        return self.padding(representation)

    def representation(self, sentence):
        '''
        Returns:
            A list of numpy arrays.
        '''
        one_hot = self.one_hot_encoding(sentence)
        return np.asarray(one_hot)

def main():
    filename = 'Data/data_augmented'
    print('before data rep')
    data_representation = DataRepresentation(filename=filename)
    print('before data feed')
    data = DataFeed(data_representation.x, data_representation.labels)
    print('after data feed')
    # Parameters
    learning_rate = 0.01
    maxiter = 10000
    batch = 128
    display = 100
    test_iter = 500

    # Network Parameters
    n_input = data_representation.size_voc # MNIST data input (img shape: 28*28)
    n_steps = data_representation.max_words # number of words
    n_hidden = 32 # hidden layer num of features
    n_classes = 6

    # Define placeholders
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    dropout = tf.placeholder("float")

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    def RNN(x, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout, output_keep_prob=dropout)
        outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

        output_layer = tf.matmul(outputs[-1], weights['out']) + biases['out']
        #return tf.nn.dropout(output_layer, keep_prob=dropout)
        return output_layer

    pred = RNN(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        for i in xrange(maxiter):
            batch_x, batch_y = data.get_next_batch(batch)
            #batch_x, batch_y = data.get_train_data()
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_y.shape[0], n_steps, n_input))
            _, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, y: batch_y, dropout:.5})
            if i % display == 0:
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            if i % test_iter == 0:
                batch_x_test, batch_y_test = data.get_test_data()
                acc_test = sess.run(accuracy, feed_dict={x:batch_x_test, y:batch_y_test, dropout:1.0})
                print("Iter %i, Testing accuracy= %.5f" % (i, acc_test))
        
        batch_x_test, batch_y_test = data.get_test_data()
        acc_test = sess.run(accuracy, feed_dict={x:batch_x_test, y:batch_y_test, dropout:1.0})
        print("Testing accuracy= %.5f" % acc_test)
        print("Optimization Finished!")

main()

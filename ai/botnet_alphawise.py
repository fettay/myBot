import os
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import preprocessing
from string import punctuation, lowercase
import argparse


def parse_flags():
    parser = argparse.ArgumentParser(prog='botnet')
    parser.add_argument('--batch', '-b', type=int, default=16, help="size of the batch for training")
    parser.add_argument('--batch_test', '-z', type=int, default=512, help="size of the batch for testing")
    #parser.add_argument('--hidden_units', '-u', nargs='+', type=int, default=[256, 128], help="number of hidden units in the LSTM cell")
    parser.add_argument('--data', '-d', type=str, required=True, help="path to the data")
    parser.add_argument('--maxiter', '-m', type=int, default=1e7, help="number of iterations for the optimization")
    parser.add_argument('--lr_init', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--lr_step', type=int, default=2000, help="decay the learning rate every x step")
    parser.add_argument('--lr_decay', type=float, default=.96, help="decay rate of the learning rate")
    parser.add_argument('--display', type=int, default=50, help="display the training metrics every x step")
    parser.add_argument('--test', '-t', type=int, default=100, help="test the network every x step")
    parser.add_argument('--prefix', '-p', type=str, required=True,
                        help="prefix for saving the model snapshots (directory also used for storing tensorboard files")
    parser.add_argument('--snapshot', '-s', type=str, default=None,
                        help="using a saved model for testing or to continue training it")
    parser.add_argument('--snapstep', type=int, default=1000, help="save a snapshot of the model every x step")
    return parser.parse_args()

class DataFeed(object):
    def __init__(self, data, labels, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=test_size)
       
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
        self.vocabulary = list(set([x for y in self.sentence_wordwise for x in y]))
        #self.max_length = min([16, max([len(x) for x in self.vocabulary])])
        self.max_length = 64
        print('max length of a sentence: ', self.max_length)
        self.alphabet = lowercase + ' ' 
        self.alphabet_n = len(self.alphabet)
        print('alphabet_n: ', self.alphabet_n)
        print('vocab length', len(self.vocabulary))
        self.one_hot_mapping = {self.alphabet[i_]: i_ for i_ in xrange(self.alphabet_n)}
        self.x = [self.representation(x) for x in self.sentence_wordwise]

    def labels_to_one_hot(self):
        self.list_labels = list(set(self.labels))
        print(self.list_labels)
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
        if len(representation) < self.max_length:
            for i in xrange(self.max_length - len(representation)):
                representation.append(np.zeros(self.alphabet_n))
        else:
            representation = representation[:self.max_length]
        return representation

    def one_hot_encoding(self, sentence):
        representation = []
        letters = ' '.join(sentence)
        for letter in letters:
            representation_ = np.zeros(self.alphabet_n)
            representation_[self.one_hot_mapping[letter]] = 1
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
    flags = parse_flags()

    tensorboard = flags.prefix[:flags.prefix.rfind('/') + 1]
    if not os.path.isdir(tensorboard):
        os.makedirs(tensorboard)
        print('%s does not exist and was created.' % tensorboard)

    data_representation = DataRepresentation(filename=flags.data)
    data = DataFeed(data_representation.x, data_representation.labels)
    # Parameters
    learning_rate = flags.lr_init
    maxiter = int(flags.maxiter)
    batch = flags.batch
    display = flags.display
    test_iter = flags.test

    # Network Parameters
    n_input = data_representation.alphabet_n
    n_steps = data_representation.max_length # number of words
    n_hidden = 32 # hidden layer num of features
    n_classes = len(data_representation.list_labels)

    # Define placeholders
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    dropout = tf.placeholder("float")

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    def botnet(x, weights, biases):
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout, output_keep_prob=dropout)

        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob=dropout, output_keep_prob=dropout)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob=dropout, output_keep_prob=dropout)
        
        outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                  dtype=tf.float32)

        #outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

        output_layer = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return tf.nn.dropout(output_layer, keep_prob=dropout)

    pred = botnet(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

# ************** Network Saver *****************
    snapshot_n = flags.snapstep
    snap_sep = '__'
    snapshot_prefix = flags.prefix + snap_sep
    pattern = r'(?:__)([0-9].*)(?:.ckpt)'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        if flags.snapshot is not None:
            print("--- Retrieving model from snapshot %s ---" %
                        flags.snapshot)
            saver.restore(sess, flags.snapshot)
            try:
                model_.iter = int(re.findall(
                    pattern, flags.snapshot)[0]) + 1
            except:
                sys.exit('Error while parsing the saved model name.')

        for i in xrange(maxiter):
            batch_x, batch_y = data.get_next_batch(batch)
            _, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, y: batch_y, dropout:.5})
            if i % display == 0:
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            if i % test_iter == 0:
                batch_x_test, batch_y_test = data.get_test_data()
                acc_test = sess.run(accuracy, feed_dict={x:batch_x_test, y:batch_y_test, dropout:1.0})
                print("Iter %i, Testing accuracy= %.5f" % (i, acc_test))
           
            if i % snapshot_n == 0:
                save_path = saver.save(
                    sess, snapshot_prefix + str(i) + '.ckpt')
                print('Snapshot saved in file: %s' % save_path)
     
        batch_x_test, batch_y_test = data.get_test_data()
        acc_test = sess.run(accuracy, feed_dict={x:batch_x_test, y:batch_y_test, dropout:1.0})
        print("Testing accuracy= %.5f" % acc_test)
        print("Optimization Finished!")

main()

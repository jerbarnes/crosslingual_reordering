from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

from Utils.WordVecs import *
from Utils.Datasets import *
from Utils.Representations import words as word_reps

from keras.models import Sequential, load_model
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Dense, Embedding, Convolution1D, MaxPooling1D, Flatten, Input, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import numpy as np
import sys
import argparse
import pickle
import json
import re

from sklearn.metrics import classification_report
from test_reordering import *


def create_cnn(matrix, max_length, dim=300, output_dim=2,
               dropout=.5, train=False):

    # Convolutional model
    filter_sizes=(2,3,4)
    num_filters = 3


    graph_in = Input(shape=(max_length, len(matrix[0])))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                 filter_length=fsz,
                 border_mode='valid',
                 activation='relu',
                 subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    out = Concatenate()(convs)
    graph = Model(input=graph_in, output=out)

    # Full model
    model = Sequential()
    model.add(Embedding(output_dim=matrix.shape[1],
                        input_dim=matrix.shape[0],
                        input_length=max_length, weights=[matrix],
                        trainable=train))
    model.add(Dropout(dropout))
    model.add(graph)
    model.add(Dense(dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='softmax'))
    if output_dim == 2:
        model.compile('adam', 'binary_crossentropy',
                  metrics=['accuracy'])
    else:
        model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_best_weights(lang, classifier='cnn', binary=False):
    if binary:
        base_dir = 'models/artetxe-'+classifier+'/binary-' + lang
    else:
        base_dir = 'models/artetxe-'+classifier+'/4class-' + lang
    weights = os.listdir(base_dir)

    best_val = 0
    best_weights = ''
    for weight in weights:
        try:
            val_f1 = re.sub('weights.[0-9]*-', '', weight)
            val_f1 = re.sub('.hdf5', '', val_f1)
            val_f1 = float(val_f1)
            if val_f1 > best_val:
                best_val = val_f1
                best_weights = weight
        except ValueError:
            pass
    return load_model(os.path.join(base_dir, best_weights), custom_objects= {'f1': f1})


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))



def add_unknown_words(wordvecs, vocab, min_fq=10, dim=50):
    """
    For words that occur less than min_fq, create a separate word vector
    0.25 is chosen so the unk vectors have approximately the same variance
    as pretrained ones
    """
    for word in vocab:
        if word not in wordvecs and vocab[word] >= min_fq:
            wordvecs[word] = np.random.uniform(-0.25, 0.25, dim)


def get_W(wordvecs, dim=300):
    """
    Returns a word matrix W where W[i] is the vector for word indexed by i
    and a word-to-index dictionary w2idx, whose keys are words and whose
    values are the indices.
    """
    vocab_size = len(wordvecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, dim), dtype='float32')

    # set unk to 0
    word_idx_map['UNK'] = 0
    W[0] = np.zeros(dim, dtype='float32')
    i = 1
    for word in wordvecs:
        W[i] = wordvecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def idx_sent(sent, w2idx):
    """
    Converts a sentence to an array of its indices
    in the word-to-index dictionary
    """
    array = []
    for w in sent:
        try:
            array.append(w2idx[w])
        except KeyError:
            array.append(0)
    return np.array(array)

def convert_dataset(dataset, w2idx, maxlen=50):
    """
    Change dataset representation from a list of lists, where each outer list
    is a sentence and each inner list contains the tokens. The result
    is a matrix of size n x m, where n is the number of sentences
    and m = maxlen is the maximum sentence size in the corpus.
    This function operates directly on the dataset and does not return any value.
    """
    dataset._Xtrain = np.array([idx_sent(s, w2idx) for s in dataset._Xtrain])
    dataset._Xdev = np.array([idx_sent(s, w2idx) for s in dataset._Xdev])
    dataset._Xtest = np.array([idx_sent(s, w2idx) for s in dataset._Xtest])
    dataset._Xtrain = pad_sequences(dataset._Xtrain, maxlen)
    dataset._Xdev = pad_sequences(dataset._Xdev, maxlen)
    dataset._Xtest = pad_sequences(dataset._Xtest, maxlen)
    return dataset

def get_projection_matrix(pdataset, src_vecs, trg_vecs):
    X, Y = [], []
    for i in pdataset._Xtrain:
        X.append(src_vecs[i])
    for i in pdataset._ytrain:
        Y.append(trg_vecs[i])

    X = np.array(X)
    Y = np.array(Y)
    u, s, vt = np.linalg.svd(np.dot(Y.T, X))
    W = np.dot(vt.T, u.T)
    return W

def str2bool(v):
    # Converts a string to a boolean, for parsing command line arguments
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', default='en', help='choose target language: en, es, ca (defaults to en)')
    parser.add_argument('-b', '--binary', default=False, help='whether to use binary or 4-class (defaults to False == 4-class)', type=str2bool)
    parser.add_argument('-se', '--src_embedding', default="embeddings/google.txt")
    parser.add_argument('-sd', '--src_dataset', default="datasets/training/en/raw")

    args = parser.parse_args()

    # Import monolingual vectors
    print('importing word embeddings')
    src_vecs = WordVecs(args.src_embedding)
    src_vecs.mean_center()
    src_vecs.normalize()

    # open datasets
    src_dataset = General_Dataset(args.src_dataset, None, rep=word_reps, binary=args.binary)
    print('src_dataset done')

    # get joint vocabulary and maximum sentence length
    print('Getting space and vocabulary...')
    max_length = 0
    src_vocab = {}
    for sentence in list(src_dataset._Xtrain) + list(src_dataset._Xdev) + list(src_dataset._Xtest):
        if len(sentence) > max_length:
            max_length = len(sentence)
        for word in sentence:
            if word in src_vocab:
                src_vocab[word] += 1
            else:
                src_vocab[word] = 1



    embeddings = {}
    for vecs in [src_vecs]:
        for w in vecs._w2idx.keys():
            # if a word is found in both source and target corpora,
            # choose the version with the highest frequency
            embeddings[w] = src_vecs[w]


    add_unknown_words(embeddings, src_vocab, min_fq=1, dim=300)
    matrix, w2idx = get_W(embeddings, dim=300)

    # save the w2idx and max length
    if args.binary:
        paramfile = 'models/artetxe-cnn/binary-{0}/{0}-w2idx.pkl'.format(args.lang)
    else:
        paramfile = 'models/artetxe-cnn/4class-{0}/{0}-w2idx.pkl'.format(args.lang)
    with open(paramfile, 'wb') as out:
        pickle.dump((w2idx, max_length), out)
    print('Saved  vocabulary...')

    # convert datasets
    src_dataset = convert_dataset(src_dataset, w2idx, max_length)

    # train BiLSTM on source
    print('Training CNN...')
    if args.binary:
        checkpoint = ModelCheckpoint('models/artetxe-cnn/binary-'+args.lang+'/weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    else:
        checkpoint = ModelCheckpoint('models/artetxe-cnn/4class-'+args.lang+'/weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    num_classes = len(set(src_dataset._ytrain.argmax(1)))
    clf = create_cnn(matrix, max_length, dim=100, output_dim=num_classes)
    history = clf.fit(src_dataset._Xtrain, src_dataset._ytrain,
                      validation_data = [src_dataset._Xdev, src_dataset._ydev],
                      verbose=1, callbacks=[checkpoint], epochs=1)

    # get the best weights to test on
    clf = get_best_weights(args.lang, classifier="cnn", binary=args.binary)

    for test_set in ["original", "random", "only_lex", "no_lex"]:
        test_directory = "datasets/{0}/{1}/raw".format(test_set, args.lang)

        test_data = TestData(test_directory, None, rep=words,
                             binary=args.binary, one_hot=True)
        print(" ".join(test_data._Xtest[0]))

        # convert dataset
        converted_test_data = convert_test_data(test_data, w2idx, max_length)

        # test classifier
        pred = clf.predict_classes(converted_test_data._Xtest)
        f1 = per_class_f1(converted_test_data._ytest.argmax(1), pred).mean()
        print(classification_report(converted_test_data._ytest.argmax(1), pred))
        print("{0} F1: {1:.3f}".format(test_set, f1))


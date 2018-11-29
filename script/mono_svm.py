from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

from Utils.WordVecs import *
from Utils.Datasets import *
from Utils.Representations import words as word_reps

from sklearn.svm import LinearSVC

import numpy as np
import sys
import argparse
import pickle
import json
import re

from sklearn.metrics import classification_report
from test_reordering import *



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

def ave_sent(sent, w2idx, matrix):
    """
    Converts a sentence to the mean
    embedding of the word vectors found
    in the sentence.
    """
    array = []
    for w in sent:
        try:
            array.append(matrix[w2idx[w]])
        except KeyError:
            array.append(np.zeros(300))
    return np.array(array).mean(0)

def convert_svm_dataset(dataset, w2idx, matrix):
    """
    Change dataset representation from a list of lists, where each outer list
    is a sentence and each inner list contains the tokens. The result
    is a matrix of size n x m, where n is the number of sentences
    and m = the dimensionality of the embeddings in the embedding matrix.
    """
    dataset._Xtrain = np.array([ave_sent(s, w2idx, matrix) for s in dataset._Xtrain])
    dataset._Xdev = np.array([ave_sent(s, w2idx, matrix) for s in dataset._Xdev])
    dataset._Xtest = np.array([ave_sent(s, w2idx, matrix) for s in dataset._Xtest])
    return dataset


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
    src_dataset = General_Dataset(args.src_dataset, None, rep=word_reps, binary=args.binary, one_hot=False)
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
        paramfile = 'models/artetxe-svm/binary-{0}/{0}-w2idx.pkl'.format(args.lang)
    else:
        paramfile = 'models/artetxe-svm/4class-{0}/{0}-w2idx.pkl'.format(args.lang)
    with open(paramfile, 'wb') as out:
        pickle.dump((w2idx, matrix), out)
    print('Saved joint vocabulary...')

    # convert datasets
    src_dataset = convert_svm_dataset(src_dataset, w2idx, matrix)


    # train Logistic Regression on source
    print('Training SVM...')
    if args.binary:
        checkpoint = 'models/artetxe-svm/binary-'+args.lang+'/weights.pkl'
    else:
        checkpoint = 'models/artetxe-svm/4class-'+args.lang+'/weights.pkl'

    num_classes = len(set(src_dataset._ytrain))
    clf = LinearSVC()
    history = clf.fit(src_dataset._Xtrain, src_dataset._ytrain)
    with open(checkpoint, 'wb') as out:
        pickle.dump(clf, out)

    for test_set in ["original", "random", "only_lex", "no_lex"]:
        test_directory = "datasets/{0}/{1}/raw".format(test_set, args.lang)

        test_data = TestData(test_directory, None, rep=words,
                             binary=args.binary, one_hot=False)
        print(" ".join(test_data._Xtest[0]))

        # convert dataset
        converted_test_data = convert_svm_test_dataset(test_data, w2idx, matrix)

        # test classifier
        pred = clf.predict(converted_test_data._Xtest)
        f1 = per_class_f1(converted_test_data._ytest, pred).mean()
        print(classification_report(converted_test_data._ytest, pred))
        print("{0} F1: {1:.3f}".format(test_set, f1))

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
    parser.add_argument('-l', '--lang', default='es',
                        help='choose target language: es, ca, eu (defaults to es)')
    parser.add_argument('-b', '--binary', default=False,
                        help='whether to use binary or 4-class (defaults to False == 4-class)',
                        type=str2bool)
    args = parser.parse_args()
    
    # Import monolingual vectors
    print('importing word embeddings')
    src_vecs = WordVecs('embeddings/google.txt')
    src_vecs.mean_center()
    src_vecs.normalize()
    trg_vecs = WordVecs('embeddings/sg-300-{0}.txt'.format(args.lang))
    trg_vecs.mean_center()
    trg_vecs.normalize()

    # Setup projection dataset
    trans = 'lexicons/bingliu_en_{0}.one-2-one.txt'.format(args.lang)
    pdataset = ProjectionDataset(trans, src_vecs, trg_vecs)

    # learn the translation matrix W
    print('Projecting src embeddings to trg space...')
    W = get_projection_matrix(pdataset, src_vecs, trg_vecs)

    # project the source matrix to the new shared space
    src_vecs._matrix = np.dot(src_vecs._matrix, W)

    # open datasets
    src_dataset = General_Dataset('datasets/en/raw', None, rep=word_reps, binary=args.binary)
    trg_dataset = General_Dataset('datasets/es/raw', None, rep=word_reps, binary=args.binary)

    # get joint vocabulary and maximum sentence length
    print('Getting joint space and vocabulary...')
    max_length = 0
    src_vocab = {}
    trg_vocab = {}
    for sentence in list(src_dataset._Xtrain) + list(src_dataset._Xdev) + list(src_dataset._Xtest):
        if len(sentence) > max_length:
            max_length = len(sentence)
        for word in sentence:
            if word in src_vocab:
                src_vocab[word] += 1
            else:
                src_vocab[word] = 1
    for sentence in list(trg_dataset._Xtrain) + list(trg_dataset._Xdev) + list(trg_dataset._Xtest):
        if len(sentence) > max_length:
            max_length = len(sentence)
        for word in sentence:
            if word in trg_vocab:
                trg_vocab[word] += 1
            else:
                trg_vocab[word] = 1
    
    
    # get joint embedding space
    joint_embeddings = {}
    for vecs in [src_vecs, trg_vecs]:
        for w in vecs._w2idx.keys():
            # if a word is found in both source and target corpora,
            # choose the version with the highest frequency
            if w in src_vocab and w in src_vecs and w in trg_vocab and w in trg_vecs:
                if src_vocab[w] >= trg_vocab[w]:
                    joint_embeddings[w] = src_vecs[w]
                else:
                    joint_embeddings[w] = trg_vecs[w]
            elif w in src_vocab and w in src_vecs:
                joint_embeddings[w] = src_vecs[w]
            elif w in trg_vocab and w in trg_vecs:
                joint_embeddings[w] = trg_vecs[w]

    joint_vocab = {}
    joint_vocab.update(src_vocab)
    joint_vocab.update(trg_vocab)
    
    add_unknown_words(joint_embeddings, joint_vocab, min_fq=1, dim=300)
    joint_matrix, joint_w2idx = get_W(joint_embeddings, dim=300)

    # save the w2idx and max length
    if args.binary:
        paramfile = 'models/artetxe-svm/binary-en-{0}/en-{0}-w2idx.pkl'.format(args.lang)
    else:
        paramfile = 'models/artetxe-svm/4class-en-{0}/en-{0}-w2idx.pkl'.format(args.lang)
    with open(paramfile, 'wb') as out:
        pickle.dump((joint_w2idx, joint_matrix), out)

    # convert datasets
    src_dataset = convert_svm_dataset(src_dataset, joint_w2idx, joint_matrix)
    trg_dataset = convert_svm_dataset(trg_dataset, joint_w2idx, joint_matrix)

    # train Logistic Regression on source
    print('Training SVM...')
    if args.binary:
        checkpoint = 'models/artetxe-svm/binary-en-'+args.lang+'/weights.pkl'
    else:
        checkpoint = 'models/artetxe-svm/4class-en-'+args.lang+'/weights.pkl'

    num_classes = len(set(src_dataset._ytrain.argmax(1)))
    clf = LinearSVC()
#    history = clf.fit(src_dataset._Xtrain, src_dataset._ytrain.argmax(1))
    history = clf.fit(trg_dataset._Xtrain, trg_dataset._ytrain.argmax(1))
    with open(checkpoint, 'wb') as out:
        pickle.dump(clf, out)

    # test on src devset and trg devset
    src_pred = clf.predict(src_dataset._Xdev)
    print(classification_report(src_dataset._ydev.argmax(1), src_pred))

    trg_pred = clf.predict(trg_dataset._Xdev)
    print(classification_report(trg_dataset._ydev.argmax(1), trg_pred))
    

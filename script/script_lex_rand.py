import nltk
import os
import argparse
from random import shuffle

#LEXICON

lexicon_path = 'lexicons/bingliu_en_es.one-2-one.txt'
lexicon_path_ca = 'lexicons/bingliu_en_ca.one-2-one.txt'

def parse_lexicon_es ():
    lex = open(lexicon_path, 'r').readlines()
    es_words = []
    for sent in lex:
        eng,es = sent.split('\t')
        es_words.append(es[:-1])
    return es_words

def parse_lexicon_ca ():
    lex = open(lexicon_path_ca, 'r').readlines()
    ca_words = []
    for sent in lex:
        eng,ca = sent.split('\t')
        ca_words.append(ca[:-1])
    return ca_words

def parse_lexicon_en ():
    lex = open(lexicon_path, 'r').readlines()
    en_words = []
    for sent in lex:
        eng,es = sent.split('\t')
        en_words.append(eng)
    return en_words
    
def modify_to_lexicon (infile, language, only_lex_no_lex):   

    new_file = []
    new_new_file = []
    lexicon = 'parse_lexicon_'+language+'()'
    if only_lex_no_lex == 'only_lex':
    
        for line in infile:
            new_sent = []
            tokens = nltk.word_tokenize(line)
            
            for t in tokens:
                if t in eval(lexicon) or t == '.':
                    new_sent.append(t)
                else:
                    new_sent.append('UNK')
            new_file.append(new_sent)
        
        for s in new_file:
            if len(s) > 0:
                if s[0] != '.':
                    new_new_file.append(s)
                            
        raw = ''
        for sent in new_new_file:
            for i in range(len(sent)):
                if i+1 != len(sent):
                    raw = raw+sent[i]+' '
                else:
                    raw = raw+sent[i]+' \n'
        
        return raw
    
    elif only_lex_no_lex == 'no_lex':
        
        for line in infile:
            new_sent = []
            tokens = nltk.word_tokenize(line)
            
            for t in tokens:
                if t not in eval(lexicon) or t == '.': #not in
                    new_sent.append(t)
                else:
                    new_sent.append('UNK')
            new_file.append(new_sent)
        
        for s in new_file:
            if len(s) > 0:
                if s[0] != '.':
                    new_new_file.append(s)
                            
        raw = ''
        for sent in new_new_file:
            for i in range(len(sent)):
                if i+1 != len(sent):
                    raw = raw+sent[i]+' '
                else:
                    raw = raw+sent[i]+' \n'
        
        return raw

def org_modify_to_lexicon (indir, language, only_lex_no_lex):
    outdir = 'new_dataset/'+indir+only_lex_no_lex+'/'
    
    if 'new_dataset/'+indir+only_lex_no_lex+'/' not in os.listdir():
        os.makedirs('new_dataset/'+indir+only_lex_no_lex+'/')
    
    for file in os.listdir(indir):
        print(file)
        with open(indir+file, 'r') as f:
            infile = f.readlines()
            outfile = outdir+file
            result = modify_to_lexicon (infile, language, only_lex_no_lex)                    
            with open(outfile, 'w') as outfile:
                outfile.write(result)
                    
#RANDOM

def random_reordering (corpus):
    new_corpus = []
    for sent in corpus:
        shuffle(sent)
        new_corpus.append(sent)
    return new_corpus

def org_random (indir):  
    outdir = 'new_dataset/'+indir+'random/'
    
    if 'new_dataset/'+indir+'random/' not in os.listdir():
        os.makedirs('new_dataset/'+indir+'random/')
        
    for file in os.listdir(indir):
        print(file)
        with open(indir+file, 'r') as f:
            corpus = []
            infile = f.readlines()
            outfile = outdir+file
            for sentence in infile:
                corpus.append(nltk.word_tokenize(sentence))
            rand = random_reordering(corpus)
            new_file = []
            for sentence in rand:
                new_sent = ' '.join(sentence)
                new_file.append(new_sent)
            with open(outfile, 'w') as outf:
                for sentence in new_file:
                    outf.write(sentence)
                    outf.write('\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset', help='dataset (directory with the txts we want to use to get its scrambled version, only_lex, and/or no_lex version')
    parser.add_argument('language', help='language of the lexicon')
    
    parser.add_argument('-r', '--random', default=False, help='reorder randomly dataset')
    parser.add_argument('-o', '--only_lex', default=False, help='substitute all words not appearing in lexicon to "UNK"')
    parser.add_argument('-n', '--no_lex', default=False, help='substitute all words that appear in lexicon to "UNK"')
    
    args = parser.parse_args()
    
    if args.only_lex in ['True', 'true']:
        org_modify_to_lexicon(args.dataset, args.language, 'only_lex')

    if args.no_lex in ['True', 'true']:
        org_modify_to_lexicon(args.dataset, args.language, 'no_lex')
    
    if args.random in ['True', 'true']:
        org_random(args.dataset)
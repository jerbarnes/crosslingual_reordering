import os
import argparse
import numpy as np

# RULES

def get_reordering_rules (mod_type="n-adj"):
    
    if mod_type == 'crego':
        
        bi_crego_rules = [[('ADJ', 'ADJ'), (1, 0)],
         [('VERB', 'PRON'), (1, 0)],
         [('ADJ', 'ADV'), (1, 0)], 
         [('NOUN', 'ADV'), (1, 0)], 
         [('NOUN', 'ADJ'), (1, 0)], 
         [('ADV', 'VERB'), (1, 0)]]

        tri_crego_rules = [[('NOUN', 'ADV', 'ADV'), (1, 2, 0)], 
         [('NOUN', 'ADV', 'ADJ'), (1, 2, 0)], 
         [('NOUN', 'ADJ', 'ADJ'), (2, 1, 0)],
         [('ADJ', 'ADV', 'ADJ'), (1, 2, 0)]]
        
        tetra_crego_rules = [[('NOUN', 'ADV', 'ADJ', 'CONJ'), (1, 2, 3, 0)],
         [('NOUN', 'ADJ', 'CONJ', 'ADJ'), (1, 2, 3, 0)],
         [('NOUN', 'CONJ', 'NOUN', 'ADJ'), (3, 0, 1, 2)], 
         [('NOUN', 'ADJ', 'ADV', 'ADJ'), (2, 3, 1, 0)]]
        
        penta_crego_rules = [[('NOUN', 'ADV', 'ADJ', 'CONJ', 'ADJ'), (1, 2, 3, 4, 0)]]
        
        crego = [penta_crego_rules, tetra_crego_rules, tri_crego_rules, bi_crego_rules]

        return crego
    
    if mod_type == 'n-adj':
        
        n_adj = [[[('.', '.', '.', '.', '.'), (1, 3, 4, 2, 0)]], [[('.', '.', '.', '.'), (1, 3, 2, 0)]], [[('.', '.', '.'), (1, 2, 0)]], [[('NOUN', 'ADJ'), (1, 0)]]]
        
        return n_adj


# TOKENIZE TAGS
        
def tokenize_tagged_file (infile):
    new_file = []
    
    with open(infile) as inf:
        for line in inf.readlines():
            sent = []
            proto = (line.split('\n')[0]).split(' ')
            for pair in proto:
                if pair:
                    if len(pair.split('/')) == 2:
                        word, tag = pair.split('/')
                        sent.append(tuple([word, tag]))
            new_file.append(sent)  
        return new_file

def delete_tags (file):
    raw = ''
    for sent in file:
        for i in range(len(sent)):
            if i+1 != len(sent):
                raw = raw+sent[i][0]+' '
            else:
                raw = raw+sent[i][0]+' \n'
    return raw

# REORDER FUNCTIONS

def reorder_bigrams (sent, set_of_rules):
    for r in set_of_rules:
        for i in range(len(sent)):
            if sent[i][1] == r[0][0]:
                if i+1<len(sent) and sent[i+1][1] == r[0][1]:
                    head = sent[:i]
                    tail = sent[i+2:]
                    body = np.array(sent[i:i+2])[list(r[1])]
                    reord_body = []
                    for element in body:
                        reord_body.append(tuple(element))
                    sent = head+reord_body+tail
    return sent

def reorder_trigrams (sent, set_of_rules):
    for r in set_of_rules:
        for i in range(len(sent)):
            if sent[i][1] == r[0][0]:
                if i+1<len(sent) and sent[i+1][1] == r[0][1]:
                    if i+2<len(sent) and sent[i+2][1] == r[0][2]:
                        head = sent[:i]
                        tail = sent[i+3:]
                        body = np.array(sent[i:i+3])[list(r[1])]
                        reord_body = []
                        for element in body:
                            reord_body.append(tuple(element))
                        sent = head+reord_body+tail
    return sent

def reorder_tetragrams (sent, set_of_rules):
    for r in set_of_rules:
        for i in range(len(sent)):
            if sent[i][1] == r[0][0]:
                if i+1<len(sent) and sent[i+1][1] == r[0][1]:
                    if i+2<len(sent) and sent[i+2][1] == r[0][2]:
                        if i+3<len(sent) and sent[i+3][1] == r[0][3]:
                            head = sent[:i]
                            tail = sent[i+4:]
                            body = np.array(sent[i:i+4])[list(r[1])]
                            reord_body = []
                            for element in body:
                                reord_body.append(tuple(element))
                            sent = head+reord_body+tail
    return sent

def reorder_pentagrams (sent, set_of_rules):
    for r in set_of_rules:
        for i in range(len(sent)):
            if sent[i][1] == r[0][0]:
                if i+1<len(sent) and sent[i+1][1] == r[0][1]:
                    if i+2<len(sent) and sent[i+2][1] == r[0][2]:
                        if i+3<len(sent) and sent[i+3][1] == r[0][3]:
                            if i+4<len(sent) and sent[i+4][1] == r[0][4]:
                                head = sent[:i]
                                tail = sent[i+5:]
                                body = np.array(sent[i:i+5])[list(r[1])]
                                reord_body = []
                                for element in body:
                                    reord_body.append(tuple(element))
                                sent = head+reord_body+tail
    return sent

def chain_reorder(corpus, ngrams_rules):
    new_corpus = []
    
    for sent in corpus:
        five = reorder_pentagrams(sent, ngrams_rules[0])
        four = reorder_tetragrams(five, ngrams_rules[1])
        three = reorder_trigrams(four, ngrams_rules[2])
        two = reorder_bigrams(three, ngrams_rules[3])
        
        new_corpus.append(two)
        
    return new_corpus

def output_reorder_raw (corpus, outfile):
    raw = delete_tags(corpus)
    with open(outfile, 'w') as outf:
        outf.write(raw)
    
    
def modify_directory(indir, outdir, mod_type="n-adj"):

    for file in os.listdir(indir):
        print(file)
        infile = os.path.join(indir, file)
        outfile = os.path.join(outdir, file)

        tokens = tokenize_tagged_file (infile)
        reordered = chain_reorder (tokens, get_reordering_rules(mod_type))
        output_reorder_raw (reordered, outfile)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_directory', help='dataset directory (directory with the POS-tagged txts we want to use to get its reordered version')
    
    parser.add_argument('output_directory', help='new dataset directory (where to print the modified versions)')
    
    parser.add_argument('-m', '--mod_type', default="n-adj", help='modification: "n-adj" reordering (default), "crego" reordering.')

    args = parser.parse_args()

    modify_directory(args.dataset_directory, args.output_directory, args.mod_type)

# NOUN-ADJ/ADJ-NOUN analysis

import os

gpath = '/home/ara/git/crosslingual_reordering/script/datasets/training/' #work directory
filenames = []
nadj_combs = [] # 978 in es. 745 in ca.
adjn_combs = [] # 23 in es (19 correctly tagged). 8 in ca (4 correctly tagged).


def get_filenames (lang):
    path ='{}/tagged/'.format(lang)
    dtsets = ['train/', 'test/', 'dev/']
    
    for d in dtsets:
        files = os.listdir(gpath+path+d)
        for f in files:
            filenames.append(gpath+path+d+f)

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

def nadj_identify (tokenized_file):
    
    # Iterate through sentences
    for i in range(len(tokenized_file)):
        sent = tokenized_file[i]
        
        # Iterate through tuples
        for j in range(len(sent)):
            tup = sent[j]
            
            if j+1 != len(sent): # omit last word in sentence
                
                if tup[1] == 'NOUN':
                    if sent[j+1][1] == 'ADJ':
                    
                        nadj_combs.append([tup, sent[j+1]])        
                    
def nadj_opposites (list_of_occurrences, tokenized_file):
    for k in range(len(list_of_occurrences)):
        tup = list_of_occurrences[k][::-1]
        
        # Iterate through sentences
        for i in range(len(tokenized_file)):
            sent = tokenized_file[i]
            
            # Iterate through tuples
            for j in range(len(sent)):
                tup2 = sent[j]
                
                if j+1 != len(sent): #omit last word in sentence
                    
                    if tup2 == tup[0]:
                        tup3 = sent[j+1]
                        if tup3 == tup[1]:
#                            print(list([tup2, tup3]), tup[::-1])
                            adjn_combs.append(list([tup2, tup3]))
#                            print('-'*20)
                


nadj_lang = get_filenames('es') # 'es' or 'ca', stores files in filenames (list)

for file in filenames:
    print('N-ADJ', str(filenames.index(file))+'/'+str(len(filenames)))
    
    # Get (NOUN, ADJ) combinations of all es or ca original files
    tokens = tokenize_tagged_file(file)
    nadj = nadj_identify(tokens) # stores in nadj_combs (list)
    print("NADJ:", len(nadj_combs))
    
for file in filenames:
    print('ADJ-N', str(filenames.index(file))+'/'+str(len(filenames)))
    
    # Get all (ADJ, NOUN) in original es or ca files that were in nadj_combs
    tokens = tokenize_tagged_file(file)
    adjn = nadj_opposites (nadj_combs, tokens) # stores in adjn_combs (list)
    print("ADJN:", len(adjn_combs))

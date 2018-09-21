#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:01:31 2018

@author: alex
"""

import nltk, re, os
from nltk import FreqDist, ngrams
from nltk import word_tokenize, sent_tokenize
#from nltk.corpus import brown
#from nltk.corpus import treebank
from prettytable import PrettyTable
import numpy as np
from collections import Counter
import shutil
from random import shuffle

from itertools import permutations

from pickle import dump, load

from nltk.tag.stanford import StanfordPOSTagger
#from nltk.tag.stanford import CoreNLPPOSTagger

# =============================================================================
# Preprocessing
# =============================================================================

os.chdir('/home/alex/Desktop/To most freq POS combs')

def get_corpus ():
    raw = open('/home/alex/Desktop/To most freq POS combs/sessions/1996-04-15.xml', 'r')
    return raw

text = get_corpus().readlines()
test = text [:290]

def enumerates ():
    enumeration = []
    for element in list(enumerate(text)):
        enumeration.append(list(element))
    return enumeration

def spanish_quotes (list):
    """This one is better because it is with the raw string"""
    new_corpus = []
    for element in list:
        regex = re.split('<[^<]+>', element)
        sentence = ''.join(regex)
        new_corpus.append(sentence)
    return new_corpus

def english_quotes (list):
    """This one is better because it is with the raw string"""
    new_corpus = []
    for element in list:
        regex = re.split('<[^<]+>', element)
        sentence = ''.join(regex)
        new_corpus.append(sentence)
    return new_corpus

def parallel ():
    #give it a session, for later
    en_corpus = []
    es_corpus = []
    for pos, line in enumerates():
        if '<p type="speech">' in line:
            index = enumerates().index([pos, line])
            previous_line = enumerates()[index-1][1]
            if '<text language="en">' in previous_line:
                line_maca1 = line.split('<p type="speech">')[1]
                english = line_maca1.split('</p>')[0]
                en_corpus.append(english)
        if '<p type="speech">' in line:
            index = enumerates().index([pos, line])
            previous_line = enumerates()[index-1][1]
            if '<text language="es">' in previous_line:
                line_maca1 = line.split('<p type="speech">')[1]
                spanish = line_maca1.split('</p>')[0]
                es_corpus.append(spanish)
    return [english_quotes(en_corpus), spanish_quotes(es_corpus)]

#testing = parallel()[1][-2]
#testeo = list(enumerate(word_tokenize(testing)))

def sent_tok ():
    en_corpus = []
    for paragraph in get_raw()[0]:
        toks = sent_tokenize(paragraph)
        en_corpus.extend(toks)
    es_corpus = []
    for paragraph in get_raw()[1]:
        toks = sent_tokenize(paragraph)
        es_corpus.extend(toks)
    return [en_corpus, es_corpus]

def tokenize (n_language):
    if n_language == 1:
        tokens = []
        for element in get_raw()[1]:
            element_tok = word_tokenize(element)
            tokens.append(element_tok)
        return tokens
    elif n_language == 0:
        tokens = []
        for element in get_raw()[0]:
            element_tok = word_tokenize(element)
            tokens.append(element_tok)
        return tokens

# =============================================================================
# TAGGER
# =============================================================================
        
jar = '/home/alex/Desktop/To most freq POS combs/stanford tagger/stanford-postagger-full-2018-02-27/stanford-postagger.jar'
model_english= '/home/alex/Desktop/To most freq POS combs/stanford tagger/stanford-postagger-full-2018-02-27/models/english-left3words-distsim.tagger'
model_spanish= '/home/alex/Desktop/To most freq POS combs/stanford tagger/stanford-postagger-full-2018-02-27/models/spanish-ud.tagger'

#stagger_en = StanfordPOSTagger(model_english, jar, encoding='utf8')
#stagger_es = StanfordPOSTagger(model_spanish, jar, encoding='utf8')

# =============================================================================
# There are _38_ unique tags in the english model.
# There are _67_ unique tags in the spaniish model (in the 'text' file, 85 total in the tagset). (or 17 in the other -universal dependencies.).
#The universal from brown has 12
# =============================================================================

def tag_english_first ():
    tagged_eng_sents = []
    stagger = StanfordPOSTagger(model_english, jar, encoding='utf8')
    for sentence in tokenize(0):
        tagged_eng_sents.append(stagger.tag(sentence))
    return tagged_eng_sents
    
def tag_spanish_first ():
    tagged_spa_sents = []
    stagger = StanfordPOSTagger(model_spanish, jar, encoding='utf8')
    for sentence in tokenize(1):
        tagged_spa_sents.append(stagger.tag(sentence))
    return tagged_spa_sents
    
def number_tags (n_language):
    if n_language == 1:
        result = []
        for sentence in tag_spanish():
            for (word, tag) in sentence:
                result.append(tag)
        return [len(set(result)), set(result)]
    elif n_language == 0:
        result = []
        for sentence in tag_english():
            for (word, tag) in sentence:
                result.append(tag)
        return [len(set(result)), set(result)]

# =============================================================================
# From ass2.
# =============================================================================
    
def freq_dist_tags (ngram, tagged_sents): #en_es_tr):
    """Calculates the frequency distribution of tag ngrams."""
    fdist = FreqDist()
    for sentence in tagged_sents:
        result = []
        for (word, tag) in sentence:
            result.append(tag)
        for element in ngrams(result, ngram, pad_left=True, pad_right=True, left_pad_symbol="$", right_pad_symbol="$"):
            if ngram==1:
                fdist[str(element[0])] +=1
            else:
                fdist[element] +=1
    return fdist

def print_common_tags_ngrams (ngram, tagged_sents, rows):
    x = PrettyTable()
    fdist = freq_dist_tags(ngram, tagged_sents)
    tags = list(tag for (tag, number) in fdist.most_common(rows))
    ngrams = []
    for a in tags:
        ngrams.append(str(a).replace('\'', ''))
    x.add_column('{}'.format(ngram) + '-gram', ngrams)
    frequencies = []
    accum_freqs = []
    for tag in tags:
        rounded_freq = str('{:.2f}'.format((fdist.freq(tag)*100)))+' %'
        frequencies.append(rounded_freq)
        tup_acc_freq = []
        for t in tags:
            if tags.index(t) <= tags.index(tag):
                tup_acc_freq.append(fdist.freq(t))
        accum_freqs.append(str('{:.2f}'.format((sum(tup_acc_freq)*100)))+' %')
    x.add_column('Frequency', frequencies)
    x.add_column('Acc. Freq.', accum_freqs)
#    os.chdir('/home/alex/Desktop/To most freq POS combs/Comparison Europarl_Tatoeeba')
#    with open('europarl_tatoeeba_comparison.txt', 'a') as file:
#        file.write(str(x))
#        file.write('\n')
    print(x)
#    os.chdir('/home/alex/Desktop/To most freq POS combs')

# =============================================================================
# TRANSLATE
# =============================================================================

def translation (english_tag):
    if english_tag == "''":
        return 'PUNCT'
    elif english_tag == ',':
        return 'PUNCT'
    elif english_tag == '.':
        return 'PUNCT'
    elif english_tag == ':':
        return 'PUNCT'
    elif english_tag == 'CC':
        return 'CONJ'
    elif english_tag == 'CD':
        return 'NUM'
    elif english_tag == 'DT':
        return 'DET'
    elif english_tag == 'EX':
        return 'PRON'
    elif english_tag == 'FW':
        return 'X'
    elif english_tag == 'IN':
        return 'ADP'
    elif english_tag == 'JJ':
        return 'ADJ'
    elif english_tag == 'JJR':
        return 'ADJ'
    elif english_tag == 'JJS':
        return 'ADJ'
    elif english_tag == 'LS':
        return 'NUM'
    elif english_tag == 'MD':
        return 'VERB'
    elif english_tag == 'NN':
        return 'NOUN'
    elif english_tag == 'NNP':
        return 'PROPN'
    elif english_tag == 'NNPS':
        return 'PROPN'
    elif english_tag == 'NNS':
        return 'NOUN'
    elif english_tag == 'PDT':
        return 'DET'
    elif english_tag == 'POS':
        return 'PART'
    elif english_tag == 'PRP':
        return 'PRON'
    elif english_tag == 'PRP$':
        return 'DET'
    elif english_tag == 'RB':
        return 'ADV'
    elif english_tag == 'RBR':
        return 'ADV'
    elif english_tag == 'RBS':
        return 'ADV'
    elif english_tag == 'RP':
        return 'ADV'
    elif english_tag == 'TO':
        return 'PART'
    elif english_tag == 'UH':
        return 'INTJ'
    elif english_tag == 'VB':
        return 'VERB'
    elif english_tag == 'VBD':
        return 'VERB'
    elif english_tag == 'VBG':
        return 'VERB'
    elif english_tag == 'VBN':
        return 'VERB'
    elif english_tag == 'VBP':
        return 'VERB'
    elif english_tag == 'VBZ':
        return 'VERB'
    elif english_tag == 'WDT':
        return 'DET'
    elif english_tag == 'WP':
        return 'PRON'
    elif english_tag == 'WRB':
        return 'ADV'
    else:
        print('Unrecognized tag.')

def better_translation (english_tag):
    if english_tag in ["''", ',', '.', ':']:
        return 'PUNCT'
    elif english_tag in ['CC']: #and, or, both, nor
        return 'CONJ'
    elif english_tag in ['CD', 'LS']: #numbers (digits and letters), lists markers.
        return 'NUM'
    elif english_tag in ['DT', 'PDT', 'PRP$']: #dets, predeterminers, possessives.
        return 'DET'
    elif english_tag in ['EX', 'MD']: #existential there -hay, modal(will, can, must).
        return 'AUX'
    elif english_tag in ['FW']: #foreign word
        return 'X'
    elif english_tag in ['IN']: #preps, sub conj (of, in, because, within, that, for, under, with, about, ...)
        return 'ADP'
    elif english_tag in ['JJ', 'JJR', 'JJS']: #adjective, comparatives, superlatives
        return 'ADJ'
    elif english_tag in ['NN', 'NNS']: #noun sing
        return 'NOUN'
    elif english_tag in ['NNP', 'NNPS']: #proper nouns
        return 'PROPN'
    elif english_tag in ['POS']: #possessive ending
        return 'PART'
    elif english_tag in ['PRP']: #pronouns
        return 'PRON'
    elif english_tag in ['RB', 'RBR', 'RBS', 'RP']: #adverbs, particles (up, out,...)
        return 'ADV'
    elif english_tag in ['TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']: #to particle, verbs.
        return 'VERB'
    elif english_tag in ['UH']: #interjections
        return 'INTJ'
    elif english_tag in ['WDT', 'WP', 'WRB']: #wh-determiner
        return 'SCONJ'

def checkt (en_es_jen_jer_jca, tag_search):
    """To check the best tag translation"""
    if en_es_jen_jer_jca == 'en':
        for sentence in get_tagged_en_first():
            for (word, tag) in sentence:
                if tag in tag_search:
                    print(word, tag)
    elif en_es_jen_jer_jca == 'es':
        for sentence in get_tagged_es_first():
            for (word, tag) in sentence:
                if tag in tag_search:
                    print(word, tag)
    elif en_es_jen_jer_jca == 'jes':
        for sentence in jer_tagged('es'):
            for (word, tag) in sentence:
                if tag in tag_search:
                    print(word, tag)
    elif en_es_jen_jer_jca == 'jen':
        for sentence in jer_tagged('en'):
            for (word, tag) in sentence:
                if tag in tag_search:
                    print(word, tag)
    elif en_es_jen_jer_jca == 'jca':
        for sentence in jer_tagged('ca'):
            for (word, tag) in sentence:
                if tag in tag_search:
                    print(word, tag)

def checkw (en_es_jen_jer_jca, word_search):
    """To check the best tag translation"""
    if en_es_jen_jer_jca == 'en':
        for sentence in get_tagged_en_first():
            for (word, tag) in sentence:
                if word in word_search:
                    print(word, tag)
    elif en_es_jen_jer_jca == 'es':
        for sentence in get_tagged_es_first():
            for (word, tag) in sentence:
                if word in word_search:
                    print(word, tag)
    elif en_es_jen_jer_jca == 'jes':
        for sentence in jer_tagged('es'):
            for (word, tag) in sentence:
                if tag in word_search:
                    print(word, tag)
    elif en_es_jen_jer_jca == 'jen':
        for sentence in jer_tagged('en'):
            for (word, tag) in sentence:
                if tag in word_search:
                    print(word, tag)
    elif en_es_jen_jer_jca == 'jca':
        for sentence in jer_tagged('ca'):
            for (word, tag) in sentence:
                if tag in word_search:
                    print(word, tag)

def translate ():
    new_corpus = []
    tagged_sents = get_tat_en_tagged()
    for s in tagged_sents:
        trans = []
        for tup in s:
            result = []
            result.extend(tup)
            result[1] = better_translation(result[1])
            new_result = tuple(result)
            trans.append(new_result)
        new_corpus.append(trans)
    return new_corpus

def compare ():
    """Compares the tagset in the translated english and translated spanish."""
    tr_tagset = []
    es_tagset = []
    for sentence in get_tagged_tr():
        for (word, tag) in sentence:
            tr_tagset.append(tag)
    for sentence in get_tagged_es():
        for (word, tag) in sentence:
            es_tagset.append(tag)
    return [set(tr_tagset), set(es_tagset)]

# =============================================================================
# Used to pickle the data.
# =============================================================================

#output_eng = open('eng_tagged_test.pkl', 'wb')
#dump(tag_english(), output_eng, -1)
#output_eng.close()
#
#output_es = open('es_tagged_test.pkl', 'wb')
#dump(tag_spanish(), output_es, -1)
#output_es.close()

#output_raw = open('raw_parallel.pkl', 'wb')
#dump(parallel(), output_raw, -1)
#output_raw.close()

#output_trans = open('eng_translated.pkl', 'wb')
#dump(translate(), output_trans, -1)
#output_trans.close()

#output_better_trans = open('eng_better_translated.pkl', 'wb')
#dump(translate(), output_better_trans, -1)
#output_better_trans.close()

#unworthy_sessions = open('unworthy.pkl', 'wb')
#dump(comparison(), unworthy_sessions, -1)
#unworthy_sessions.close()
    
#full_corpus = open('full_corpus.pkl', 'wb')
#dump(corpus(), full_corpus, -1)
#full_corpus.close()

#tokens = open('tokenized.pkl', 'wb')
#dump(tokenization(), tokens, -1)
#tokens.close()

#tokens_en = open('tokenized_en.pkl', 'wb')
#dump(tokenization()[0], tokens_en, -1)
#tokens_en.close()

#tokens_es = open('tokenized_es.pkl', 'wb')
#dump(tokenization()[1], tokens_es, -1)
#tokens_es.close()

#out_eng = open('eng_tagged.pkl', 'wb')
#dump(tag_english(), out_eng, -1)
#out_eng.close()

#out_es = open('es_tagged.pkl', 'wb')
#dump(tag_spanish(), out_es, -1)
#out_es.close()

#out_es_4000 = open('es_tagged_4000.pkl', 'wb')
#dump(tag_spanish(), out_es_4000, -1)
#out_es_4000.close()

#out_en_4000 = open('en_tagged_4000.pkl', 'wb')
#dump(tag_english(), out_en_4000, -1)
#out_en_4000.close()
    
#output_better_trans_full_4000 = open('eng_better_translated_full_4000.pkl', 'wb')
#dump(translate(), output_better_trans_full_4000, -1)
#output_better_trans_full_4000.close()

#fus_es = open('es_fusioned.pkl', 'wb')
#dump(fusion_verbs(get_tagged_es()), fus_es, -1)
#fus_es.close()

#fus_tr = open('tr_fusioned.pkl', 'wb')
#dump(fusion_verbs(get_tagged_tr()), fus_tr, -1)
#fus_tr.close()
    
# =============================================================================
# Loading pickled data
# =============================================================================

def get_raw ():
    """Gets the pickled raw parallel data."""
    input_raw = open('raw_parallel.pkl', 'rb')
    return load(input_raw)

def get_tagged_es_first ():
    """Gets the pickled first session tagged in spanish"""
    input_es = open('es_tagged_test.pkl', 'rb')
    return load(input_es)

def get_tagged_en_first ():
    """Gets the pickled first session tagged in english"""
    input_eng = open('eng_tagged_test.pkl', 'rb')
    return load(input_eng)

def get_tagged_tr_old ():
    """Gets the pickled first session tagged in english OLD tagged translated"""
    input_eng = open('eng_translated.pkl', 'rb')
    return load(input_eng)

def get_tagged_tr_first ():
    """Gets the pickled first session tagged in english tagged translated"""
    input_eng = open('eng_better_translated.pkl', 'rb')
    return load(input_eng)

# =============================================================================
# TO ANALISE THE WHOLE CORPUS

# =============================================================================
# EXTRACTION PARALLEL
# =============================================================================

def extract_parallel_corpus ():
    os.chdir('/home/alex/Desktop/To most freq POS combs')
    path = '/home/alex/Desktop/To most freq POS combs/sessions/'
    for file in os.listdir(path):
        filename = path + file
        raw = open(filename, 'r')
        enumeration = []
        for element in list(enumerate(raw.readlines())):
            enumeration.append(list(element))
        en_corpus = []
        es_corpus = []
        for pos, line in enumeration:
            if '<p type="speech">' in line:
                index = enumeration.index([pos, line])
                previous_line = enumeration[index-1][1]
                if '<text language="en">' in previous_line:
                    line_maca1 = line.split('<p type="speech">')[1]
                    english = line_maca1.split('</p>')[0]
                    en_corpus.append(english)
            if '<p type="speech">' in line:
                index = enumeration.index([pos, line])
                previous_line = enumeration[index-1][1]
                if '<text language="es">' in previous_line:
                    line_maca1 = line.split('<p type="speech">')[1]
                    spanish = line_maca1.split('</p>')[0]
                    es_corpus.append(spanish)
        os.chdir('/home/alex/Desktop/To most freq POS combs/RAW PICKLED')
        output_parallel = open(re.sub('.xml', '.pkl', file), 'wb')
        dump([en_corpus, es_corpus], output_parallel, -1)
        output_parallel.close()

# =============================================================================
# QUOTES AND TOKENIZATION
# =============================================================================
        
def clean_quotes (llista):
    new_corpus = []
    for element in llista:
        regex = re.split('<[^<]+>', element)
        sentence = ''.join(regex)
        new_corpus.append(sentence)
    return new_corpus

def comparison ():
    pickles_path = '/home/alex/Desktop/To most freq POS combs/RAW PICKLED/'
    empty = []
    difference = []
    for i in range(0, len(os.listdir(pickles_path))):
        en_len = len(get_raw_session(i)[0])
        es_len = len(get_raw_session(i)[1])
        if en_len != es_len:
            if en_len == 0 or es_len == 0:
                empty.append(i)
            if abs(en_len-es_len) > 0:
                difference.append(i)
    unworthy = list(set(empty+difference))
    return unworthy

def corpus ():
    """Puts together all the wanted raw parallel data in one corpus."""
    pickles_path = '/home/alex/Desktop/To most freq POS combs/RAW PICKLED/'
    en_corp = []
    es_corp = []
    for i in range(0, len(os.listdir(pickles_path))):
        if i not in get_unworthy():
            en_corp.append(get_raw_session(i)[0])
            es_corp.append(get_raw_session(i)[1])
    return en_corp, es_corp

def tokenization ():
    """And clean quotes."""
    en_quotes = []
    es_quotes = []
    for session in get_parallel()[0]:
            en_quotes.append(clean_quotes(session))
    for session in get_parallel()[1]:
            es_quotes.append(clean_quotes(session))
    en_sents = []
    es_sents = []
    for session in en_quotes:
        for paragraph in session:
            en_sents.extend(sent_tokenize(paragraph))
    for session in es_quotes:
        for paragraph in session:
            es_sents.extend(sent_tokenize(paragraph))
    en_corpus = []
    es_corpus = []
    for sentence in en_sents:
        en_corpus.append(word_tokenize(sentence))
    for sentence in es_sents:
        es_corpus.append(word_tokenize(sentence))
    return en_corpus, es_corpus

def enum_tokens_en ():
    enumeration = []
    for element in list(enumerate(get_tokens_en()[:1500])):
        enumeration.append(list(element))
    return enumeration

def enum_tokens_es ():
    enumeration = []
    for element in list(enumerate(get_tokens_es()[:1500])):
        enumeration.append(list(element))
    return enumeration

def tokenize_words (tokenized_sents):
    result = []
    for sent in tokenized_sents:
        result.extend(sent)
    return result

# =============================================================================
# TAGGING: get_tokens_es()[:1430] // get_tokens_en()[:1416])
# =============================================================================

def tag_english ():
    tagged_eng_sents = []
    stagger = StanfordPOSTagger(model_english, jar, encoding='utf8')
    for sentence in get_tat_en_tok():
        tagged_eng_sents.append(stagger.tag(sentence))
    return tagged_eng_sents
    
def tag_spanish ():
    tagged_spa_sents = []
    stagger = StanfordPOSTagger(model_spanish, jar, encoding='utf8')
    for sentence in get_tat_es_tok():
        tagged_spa_sents.append(stagger.tag(sentence))
    return tagged_spa_sents


# =============================================================================
# LOAD PICKLED DATA FOR WHOLE CORPUS
# =============================================================================

def get_raw_session (session_number):
    pickles_path = '/home/alex/Desktop/To most freq POS combs/RAW PICKLED/'
    files = sorted(os.listdir('/home/alex/Desktop/To most freq POS combs/RAW PICKLED/'))
    session = open(pickles_path+files[session_number], 'rb')
    return load(session)

def get_parallel ():
    par = open('full_corpus.pkl', 'rb')
    return load(par)

def get_unworthy ():
    unw = open('unworthy.pkl', 'rb')
    return load(unw)

def get_tokens_en ():
    tok = open('tokenized_en.pkl', 'rb')
    return load(tok)

def get_tokens_es ():
    tok = open('tokenized_es.pkl', 'rb')
    return load(tok)

def get_tokens ():
    tok = open('tokenized.pkl', 'rb')
    return load(tok)

def get_tagged_es_500 ():
    input_es = open('es_tagged.pkl', 'rb')
    return load(input_es)

def get_tagged_en_500 ():
    input_eng = open('eng_tagged.pkl', 'rb')
    return load(input_eng)

def get_tagged_tr_500 ():
    input_eng = open('eng_better_translated_full.pkl', 'rb')
    return load(input_eng)

def get_tagged_es ():
    input_es = open('es_tagged_4000.pkl', 'rb')
    return load(input_es)

def get_tagged_en ():
    input_eng = open('en_tagged_4000.pkl', 'rb')
    return load(input_eng)

def get_tagged_tr ():
    input_eng = open('eng_better_translated_full_4000.pkl', 'rb')
    return load(input_eng)

def get_fusioned_es ():
    inp = open('es_fusioned.pkl', 'rb')
    return load(inp)

def get_fusioned_tr ():
    inp = open('tr_fusioned.pkl', 'rb')
    return load(inp)

# =============================================================================
# To polish list
# =============================================================================

def compare_verbs_sequences (ngram, tagged_sents):
    fdist = freq_dist_tags(ngram, tagged_sents)
    verb = 'VERB'
    for element in fdist:
        if ngram == 1:
            if element == verb:
                return element, fdist[element]
        elif ngram == 2:
            if element[0] == verb and element[1] == verb:
                    return element, fdist[element]
        elif ngram == 3:
            if element[0] == verb and element[1] == verb:
                if element[2] == verb:
                    return element, fdist[element]
        elif ngram == 4:
            if element[0] == verb and element[1] == verb:
                if element[2] == verb and element[3] == verb:
                    return element, fdist[element]
        elif ngram == 5:
            if element[0] == verb and element[1] == verb:
                if element[2] == verb and element[3] == verb:
                    if element[4] == verb:
                        return [element, fdist[element]]
        elif ngram == 6:
            if element[0] == verb and element[1] == verb:
                if element[2] == verb and element[3] == verb:
                    if element[4] == verb and element[5] == verb:
                        return [element, fdist[element]]
        elif ngram == 7:
            if element[0] == verb and element[1] == verb:
                if element[2] == verb and element[3] == verb:
                    if element[4] == verb and element[5] == verb:
                        if element[6] == verb:
                            return [element, fdist[element]]

def verbs_table ():
    """Prints the relevant info regarding v+v(...) combination."""
    x = PrettyTable()
    x.field_names = ['Language', 'Tag comb.', 'Freq']
    for i in range(1, 4):
        tags = compare_verbs_sequences(i, get_tagged_es())[0]
        for element in compare_verbs_sequences(i, get_tagged_es()):
            freq = round(freq_dist_tags(i, get_tagged_es()).freq(compare_verbs_sequences(i, get_tagged_es())[0])*100,4)
            strfreq = str(freq) + ' %'
        x.add_row(['es', tags, strfreq])
    x.add_row(['-'*3, '-'*20, '-'*3])
    for i in range(1, 6):
        tags = compare_verbs_sequences(i, get_tagged_tr())[0]
        for element in compare_verbs_sequences(i, get_tagged_tr()):
            freq = round(freq_dist_tags(i, get_tagged_tr()).freq(compare_verbs_sequences(i, get_tagged_tr())[0])*100,4)
            strfreq = str(freq) +' %'
        x.add_row(['tr', tags, strfreq])
    x.add_row(['-'*3, '-'*20, '-'*3])
    es_fus_tags = compare_verbs_sequences(1, get_fusioned_es())[0]
    tr_fus_tags = compare_verbs_sequences(1, get_fusioned_tr())[0]
    es_fus_freq = round(freq_dist_tags(1, get_fusioned_es()).freq(compare_verbs_sequences(1, get_fusioned_es())[0])*100,4)
    es_fus_strfreq = str(es_fus_freq) +' %'
    tr_fus_freq = round(freq_dist_tags(1, get_fusioned_tr()).freq(compare_verbs_sequences(1, get_fusioned_tr())[0])*100,4)
    tr_fus_strfreq = str(tr_fus_freq) +' %'
    x.add_row(['es_fus', es_fus_tags, es_fus_strfreq])
    x.add_row(['-'*3, '-'*20, '-'*3])
    x.add_row(['tr_fus', tr_fus_tags, tr_fus_strfreq])
    print(x)

def get_examples_verbs ():
    enumeration = []
    verbs = []
    for sentence in get_tagged_tr():
        enumeration.append(list(enumerate(sentence)))
    for sentence in enumeration:
        for (pos, (word, tag)) in sentence:
            tup = (pos, (word, tag))
            previous_pos = sentence.index((pos, (word, tag)))-1
            previous_tuple = sentence[previous_pos]
            if tag == 'VERB' and previous_tuple[-1][-1] == 'VERB':
                verbs.append(sentence [previous_pos-3])
                verbs.append(sentence [previous_pos-2])
                verbs.append(sentence [previous_pos-1])
                verbs.append(previous_tuple)
                verbs.append(tup)
                verbs.append(sentence [previous_pos+2])
                verbs.append('-'*20)
    return verbs
        
def get_slices (tagged_sents):
    corpus_verbs = []
    good_corpus_verbs = []
    for sentence in tagged_sents:
        sent_enumeration = []
        sent_enumeration.extend(list(enumerate(sentence)))
        sent_verbs = []
        for pos, tupl in sent_enumeration:
            tag = tupl[-1]
            if tag == 'VERB':
                sent_verbs.append(pos)
                for i in range(pos+1, len(sent_enumeration)):
                    next_tag = sent_enumeration[i][-1][-1]
                    if next_tag == 'VERB':
                        new_pos = sent_enumeration[i][0]
                        if new_pos-1 in sent_verbs:
                            sent_verbs.append(new_pos)
        nice_sent_verbs = sorted(set(sent_verbs))
        corpus_verbs.append(nice_sent_verbs)
    for lista in corpus_verbs:
        good_lista = []
        for position in lista:
            if position-1 in lista:
                good_lista.append(position)
            if position+1 in lista:
                good_lista.append(position)
        good_corpus_verbs.append(sorted(set(good_lista)))
    return good_corpus_verbs          
       
def enum_verbs (tagged_sents):
    enumeration_sents = []
    for element in list(enumerate(tagged_sents)):
        enumeration_sents.append(element)
    enumeration_slices = []
    for element in list(enumerate(get_slices(tagged_sents))):
        enumeration_slices.append(element)
    return enumeration_sents, enumeration_slices

def head_tails (list_of_integers):
    list2 = []
    for element in list_of_integers:
        if element+1 not in list_of_integers:
            list2.append(element)
        elif element-1 not in list_of_integers:
            list2.append(element)
    return list2

def fusion_verbs (tagged_sents):
    unpacked_corpus = []
    corpus = []
    for pos, slices in enum_verbs(tagged_sents)[-1]:
        for pos2, sent in enum_verbs(tagged_sents)[0]:
            if pos == pos2:
#                print(head_tails(slices))
                if len(head_tails(slices)) > 0:
                    head = head_tails(slices)[0]+1
                    tail = head_tails(slices)[-1]+1
                    bigs = sorted(nltk.bigrams(head_tails(slices)[1:-1]))[::2]
                    body = []
                    for combination in bigs:
                        new_comb = []
                        for num in combination:
                            new_comb.append(num+1)
                        body.append(new_comb)
                    head_sent = tagged_sents[pos][:head]
                    body_sent = []
                    if len(body) == 0:
                        whole_sent = head_sent+tagged_sents[pos][tail:]
                        corpus.append([pos, whole_sent])
                    else:
                        for el in body:
                            body_slicing = tagged_sents[pos][el[0]:el[1]]
                            body_sent.append(body_slicing)
                        if len(body_sent) != 0:
                            whole_b = []
                            for s in body_sent:
                                whole_b.extend(s)
                            tail_sent = tagged_sents[pos][tail:]
                            whole_s = head_sent+whole_b+tail_sent
                            corpus.append([pos, whole_s])
                else:
                    corpus.append([pos, sent])
    for position, sentence in corpus:
        unpacked_corpus.append(sentence)
    return unpacked_corpus
                        
        
# =============================================================================
# TATOEEBA
# =============================================================================

def tat_extract ():
    tat_path = '/home/alex/Desktop/To most freq POS combs/INFORMAL CORPORA/TATOEEBA/jeremy/En-Es-tatoeba.txt'
    raw = open(tat_path, 'r')
    return raw
    
def tat_tokenize ():
    """And clean apostrophes. \t is used for different translations. || divides languages. \n divides sentences."""
    clean = []
    es_chunks = []
    en_chunks = []
    es_sents = []
    en_sents = []
    es_tokens = []
    en_tokens = []
    for chunk in tat_extract().readlines()[:4000]:
        clean.append(re.sub('&#039;', "'", chunk))
    for clean_chunk in clean:
        es_chunks.append(clean_chunk.split("||")[1])
        en_chunks.append(clean_chunk.split("||")[0])
    for chunk in en_chunks:
        if len(chunk.split('\t')[0]) != 0:
            en_sents.append(chunk.split('\t')[0])
        else:
            en_sents.append(chunk.split('\t')[1])
    for chunk in es_chunks:
        if len(chunk.split('\t')[0]) != 0:
            es_sents.append(chunk.split('\t')[0])
        else:
            es_sents.append(chunk.split('\t')[1])
    for sent in en_sents:
        en_tokens.append(word_tokenize(sent))
    for sent in es_sents:
        es_tokens.append(word_tokenize(sent))
    return en_tokens, es_tokens


# =============================================================================
# PICKLE THE DATA
# =============================================================================

#tat_es_tok = open('tat_es_tok.pkl', 'wb')
#dump(tat_tokenize()[1], tat_es_tok, -1)
#tat_es_tok.close()

#tat_en_tok = open('tat_en_tok.pkl', 'wb')
#dump(tat_tokenize()[0], tat_en_tok, -1)
#tat_en_tok.close()

#tat_en_tagged = open('tat_en_tagged.pkl', 'wb')
#dump(tag_english(), tat_en_tagged, -1)
#tat_en_tagged.close()

#tat_es_tagged = open('tat_es_tagged.pkl', 'wb')
#dump(tag_spanish(), tat_es_tagged, -1)
#tat_es_tagged.close()

#tat_tr_tagged = open('tat_tr_tagged.pkl', 'wb')
#dump(translate(), tat_tr_tagged, -1)
#tat_tr_tagged.close()

# =============================================================================
# LOAD PICKLED DATA
# =============================================================================

def get_tat_es_tok ():
    inp = open('tat_es_tok.pkl', 'rb')
    return load(inp)

def get_tat_en_tok ():
    inp = open('tat_en_tok.pkl', 'rb')
    return load(inp)

def get_tat_en_tagged ():
    inp = open('tat_en_tagged.pkl', 'rb')
    return load(inp)

def get_tat_es_tagged ():
    inp = open('tat_es_tagged.pkl', 'rb')
    return load(inp)

def get_tat_tr_tagged ():
    inp = open('tat_tr_tagged.pkl', 'rb')
    return load(inp)

# =============================================================================
# CREGO AND MARIÑO PAPERS
# =============================================================================

def tokenize_lists_crego ():
    raw = open('/home/alex/Desktop/To most freq POS combs/Llistes Crego i Mariño.txt', 'r').readlines()
    first_chunk = raw[14:26]
    second_chunk = raw[31:-5]
    first_list = []
    second_list = []
    for line in first_chunk[:6]:
        first_list.append(line.split(';')[0])
    for line in second_chunk:
        second_list.append(line.split(';')[0])
    together = first_list+second_list
    final_list = []
    for element in together:
        final_list.append(element[:-1])
    return final_list

def get_crego_tagset ():
    tagset = []
    for line in list(set(tokenize_lists_crego())):
        for tag in line.split(' '):
            tagset.append(tag)
    return sorted(set(tagset))

def crego_translation (crego_tag):
    #It uses the freeling tagset, but only the first two letters of every tag.
    if crego_tag in ['AQ']: #adjectives
        return 'ADJ'
    elif crego_tag in ['CC']: #and, or, both, nor
        return 'CONJ'
    elif crego_tag in ['NC']: #nouns
        return 'NOUN'
    elif crego_tag in ['PP']: #pronouns
        return 'PRON'
    elif crego_tag in ['RG', 'RN']: #adverbs
        return 'ADV'
    elif crego_tag in ['VA', 'VM', 'VS']: #verbs
        return 'VERB'

def get_reorderings_crego ():
    raw = open('/home/alex/Desktop/To most freq POS combs/Llistes Crego i Mariño.txt', 'r').readlines()
    first_chunk = raw[14:26]
    second_chunk = raw[31:-5]
    first_list = []
    second_list = []
    reorderings = []
    regex = []
    for line in first_chunk[:6]:
        first_list.append(line.split(';')[1][1:])
    for line in second_chunk:
        second_list.append(line.split(';')[1][1:])
    together = first_list+second_list
    for line in together:
        regex.append(re.findall(r'[\d]+', line))
    for line in regex:
        rule = []
        for num in line:
            rule.append(int(num))
        reorderings.append(rule)
    return reorderings

def get_rules_crego ():
    """And translate the tagset. An set(). Rules are es -> en. For example:
    ('Control fronterizo más estricto')->('Más estricto fronterizo control').
    ('NOUN', 'ADJ', 'ADV', 'ADJ') -> (2 3 1 0)."""
    rules = []
    all_rules = []
    for line in tokenize_lists_crego():
        trans_rules = []
        for tag in line.split(' '):
            trans_rules.append(crego_translation(tag))
        all_rules.append(trans_rules)
    for i in range(0, len(get_reorderings_crego())):
        rule = []
        rule.append(all_rules[i])
        rule.append(get_reorderings_crego()[i])
        rules.append(rule)
    to_set_full = []
    for line in rules:
        to_set = []
        for el in line:
            to_set.extend(el)
        to_set_full.append(tuple(to_set))
#    return list(set(to_set_full))
    final = []
    for element in list(set(to_set_full)):
        result = []
        tag_combs = element[:int(len(element)/2)]
        reords = element[int(len(element)/2):]
        result.append(tuple(list(tag_combs)))
        combs = []
        for number in reords:
            combs.append(tag_combs[number])
        result.append(tuple(combs))
        final.append(result)
    return final

def get_rules_crego_new ():
    rules = [[('NOUN', 'ADV', 'ADJ', 'CONJ'), (1, 2, 3, 0)],
 [('NOUN', 'ADV', 'ADV'), (1, 2, 0)],
 [('NOUN', 'ADV', 'ADJ'), (1, 2, 0)],
 [('ADJ', 'ADJ'), (1, 0)],
 [('VERB', 'PRON'), (1, 0)],
 [('NOUN', 'ADJ', 'CONJ', 'ADJ'), (1, 2, 3, 0)],
 [('NOUN', 'CONJ', 'NOUN', 'ADJ'), (3, 0, 1, 2)],
 [('ADJ', 'ADV'), (1, 0)],
 [('NOUN', 'ADV', 'ADJ', 'CONJ', 'ADJ'), (1, 2, 3, 4, 0)],
 [('NOUN', 'ADJ', 'ADJ'), (2, 1, 0)],
 [('NOUN', 'ADV'), (1, 0)],
 [('ADJ', 'ADV', 'ADJ'), (1, 2, 0)],
 [('NOUN', 'ADJ'), (1, 0)],
 [('ADV', 'VERB'), (1, 0)],
 [('NOUN', 'ADJ', 'ADV', 'ADJ'), (2, 3, 1, 0)]]
    return rules

def print_rules_crego ():
    from prettytable import PrettyTable
    x = PrettyTable()
    x.field_names = ['ES Tags', 'Tag Trans']
    for element in get_rules_crego():
        x.add_row(element)
    print(x)

def organize_crego_rules ():
    """[ngram][combination][spanish or translation to english][]"""
    bi = []
    tri = []
    tetra = []
    penta = []
    for element in get_rules_crego():
        ngram = len(element[0])
        if ngram == 2:
            bi.append(element)
        elif ngram == 3:
            tri.append(element)
        elif ngram == 4:
            tetra.append(element)
        elif ngram == 5:
            penta.append(element)
    return bi, tri, tetra, penta

def check_crego_rules (tagged_sents):
    """Checks whether Crego's combs are in my analysis."""
    for element in get_rules_crego():
        ngram_len = len(element[0])
        freq = round(freq_dist_tags(ngram_len, tagged_sents).freq(element[0])*100,4)
        print (str(freq), '%')

def print_crego_freqs_en (tagged_sents):
    x = PrettyTable()
    x.field_names = ['POS', 'Frequency', 'Accumulated Frequency']
    pecking_order = []
    for el in get_rules_crego():
        frequency = round(freq_dist_tags(len(el[0]), tagged_sents).freq(el[1])*100,3)
        pecking_order.append([el[1], frequency])
    pos_combs = sorted(pecking_order, key = lambda x: float(x[1]), reverse=True)
    for element in pos_combs:
        result = []
        frequency = str(element[1]) + ' %'
        accum_freq = []
        for c in pos_combs:
            if pos_combs.index(c) <= pos_combs.index(element):
                accum_freq.append(c[1])
        result.append(element[0])
        result.append(frequency)
        result.append(str(round(sum(accum_freq),3))+' %')
        x.add_row(result)
    print(x)

def print_crego_freqs_es (tagged_sents):
    x = PrettyTable()
    x.field_names = ['POS', 'Frequency', 'Accumulated Frequency']
    pecking_order = []
    for el in get_rules_crego():
        frequency = round(freq_dist_tags(len(el[0]), tagged_sents).freq(el[0])*100,3)
        pecking_order.append([el[0], frequency])
    pos_combs = sorted(pecking_order, key = lambda x: float(x[1]), reverse=True)
    for element in pos_combs:
        result = []
        frequency = str(element[1]) + ' %'
        accum_freq = []
        for c in pos_combs:
            if pos_combs.index(c) <= pos_combs.index(element):
                accum_freq.append(c[1])
        result.append(element[0])
        result.append(frequency)
        result.append(str(round(sum(accum_freq),3))+' %')
        x.add_row(result)
    print(x)

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

# =============================================================================
# GET CREGO RULES SET
# =============================================================================
    
def check_crego_pos_en_tatoeeba ():
    result = []
    not_here = []
    for rule in get_rules_crego():
        fdist = freq_dist_tags(len(rule[1]), get_tat_tr_tagged())
        for element in fdist.most_common():
            if rule[1] == element[0]:
                result.append(rule[1])
        if rule[1] not in result:
            not_here.append(rule[1])
    return not_here

def check_crego_pos_es_tatoeeba ():
    result = []
    not_here = []
    for rule in get_rules_crego():
#        rules_combs.append(rule[1])
        fdist = freq_dist_tags(len(rule[0]), get_tat_es_tagged())
        for element in fdist.most_common():
            if rule[0] == element[0]:
                result.append(rule[0])
        if rule[0] not in result:
            not_here.append(rule[0])
    return not_here

def check_lowest_crego_rule (tagged_sents):
    """Two changes from 0 to 1 or viceversa if english or spanish.
    Displays the % in which they occur in the most_common()"""
    result = []
    for rule in get_rules_crego():
        fdistmc = freq_dist_tags(len(rule[0]), tagged_sents).most_common()
        for element in fdistmc:
            if rule[0] == element[0]:
                pos = fdistmc.index(element)
                num = round(((pos/len(fdistmc)) * 100),2)
                result.append([rule[0], num])
    return sorted(result, key=lambda x: x[-1])

def check_crego_dist (tagged_sents_es, tagged_sents_en):
    """Check the dist in pos between half of the rule and the other in their respective languages corpora."""
    result = []
    difference = []
    for rule in get_rules_crego():
        rule_result = []
        rule_result.append(rule)
        fdistmc1 = freq_dist_tags(len(rule[0]), tagged_sents_es).most_common()
        fdistmc2 = freq_dist_tags(len(rule[1]), tagged_sents_en).most_common()
        total = [len(fdistmc1), len(fdistmc2)]
        rule_result.append(total)
        for element in fdistmc1:
            if rule[0] == element[0]:
                pos1 = fdistmc1.index(element)
                rule_result.append(pos1)
        for element in fdistmc2:
            if rule[1] == element[0]:
                pos2 = fdistmc2.index(element)
                rule_result.append(pos2)
        result.append(rule_result)
#    return result
    for element in result:
        es_freq = round((element[-2]/element[-3][0]*100),4)
        en_freq = round((element[-1]/element[-3][1]*100),4)
        diff = round((abs(es_freq - en_freq)),2)
        difference.append([element[0], diff])
    return sorted(difference, key=lambda x : x[-1])
    
# =============================================================================
# GET EUROPARL RULES SET
# =============================================================================

#EXTRACTION WITH "NO CRITERIA" (orders them by pos, and es_pos1 -> en_pos1).

def OLD_rules_eu_nocr ():
    es = get_fusioned_es()
    en = get_fusioned_tr()
    rules = []
    final_rules = []
    for ngram in range(2,3):#5,6):#6):
        fdistmc_es = freq_dist_tags(ngram, es).most_common()
        fdistmc_en = freq_dist_tags(ngram, en).most_common()
        for tag_comb_es in fdistmc_es:
            rule = []
            possible = sorted(permutations(tag_comb_es[0], ngram))
            pos_rule_en = []
            for element in possible:
                for tag_comb_en in fdistmc_en:
                    if tag_comb_en[0] != tag_comb_es[0]:
                        if element == tag_comb_en[0]:
                            pos_rule_en.append(tag_comb_en)
            final = sorted(pos_rule_en, key=lambda x : x[-1], reverse=True)
            if final :
                rule.append([tag_comb_es[0], final[0][0]])
            rules.append(rule)
    for element in rules:
        if element:
            final_rules.extend(element)
    return final_rules


#EXTRACTION WITH CRITERIA 1 (don't take the combs that are too uncommon in the freqs of their own ngram (and language, ofc))

def rules_eu_crone (ngram):
    """Two changes from 0 to 1 or viceversa if english or spanish."""
    result = []
    final_result = []
    final_rules = []
    limit = 5 #48
    for rule in get_eu_nocr_rules(ngram):
        fdistmc = freq_dist_tags(ngram, get_fusioned_es()).most_common()
        for element in fdistmc:
            if rule[0] == element[0]:
                pos = fdistmc.index(element)
                num = round(((pos/len(fdistmc)) * 100),2)
                result.append([rule[0], num])
    sorted_result = sorted(result, key=lambda x: x[-1])
    for element in sorted_result:
        if element[-1] != 0:
            if element[-1] < limit:
                final_result.append(element)
#    return final_result
    for element in final_result:
        for rule in get_eu_nocr_rules(ngram):
            if element[0] == rule[0]:
                final_rules.append(rule)
    return final_rules

def check_overlapping_rules(ngram):
    result = []
    for rule in get_eu_nocr_rules(ngram):
        for rule2 in get_eu_nocr_rules(ngram):
            if rule != rule2:
                if rule[0] == rule2[0]:
                    result.append([rule, rule2])
    return result

#EXTRACTION WITH CRITERIA 2 (don't make rules if one part of the rule is very common and the other is too uncommon in their own corpora respectively)

def check_eu_dist (rules_set, tagged_sents_es, tagged_sents_en):
    """ALSO FOR TATOEEBA. Check the dist in pos between half of the rule and the other in their respective languages corpora."""
    result = []
    difference = []
    for rule in rules_set:
        rule_result = []
        rule_result.append(rule)
        fdistmc1 = freq_dist_tags(len(rule[0]), tagged_sents_es).most_common()
        fdistmc2 = freq_dist_tags(len(rule[1]), tagged_sents_en).most_common()
        total = [len(fdistmc1), len(fdistmc2)]
        rule_result.append(total)
        for element in fdistmc1:
            if rule[0] == element[0]:
                pos1 = fdistmc1.index(element)
                rule_result.append(pos1)
        for element in fdistmc2:
            if rule[1] == element[0]:
                pos2 = fdistmc2.index(element)
                rule_result.append(pos2)
        result.append(rule_result)
#    return result
    for element in result:
        if len(element) != 4:
            while len(element) < 4:
                element.append(min(element[1]))
        es_freq = round((element[-2]/element[-3][0]*100),4)
        en_freq = round((element[-1]/element[-3][1]*100),4)
        diff = round((abs(es_freq - en_freq)),2)
        difference.append([element[0], diff])
    return sorted(difference, key=lambda x : x[-1])

def filter_disparity (ngram, check_dist):
    limit = 10 #68.54
#    limit = 1
    result = []
    for element in check_dist(ngram):
        if element[-1] < limit:
            result.append(element[0])
    return result
        

#EXTRACTION WITH BOTH CRITERIA (as long as it is the same set, it is a rule)

"""filter_disparity with get_check_eu_both_rules"""


# =============================================================================
# GET TATOEEBA RULES SET
# =============================================================================

#EXTRACTION WITH "NO CRITERIA" (orders them by pos, and es_pos1 -> en_pos1).
    
def OLD_rules_tat_nocr ():
    es = get_tat_es_tagged()
    en = get_tat_tr_tagged()
    rules = []
    final_rules = []
    for ngram in range(2, 3):
        fdistmc_es = freq_dist_tags(ngram, es).most_common()
        fdistmc_en = freq_dist_tags(ngram, en).most_common()
        for tag_comb_es in fdistmc_es:
            rule = []
            possible = sorted(permutations(tag_comb_es[0], ngram))
            pos_rule_en = []
            for element in possible:
                for tag_comb_en in fdistmc_en:
                    if tag_comb_en[0] != tag_comb_es[0]:
                        if element == tag_comb_en[0]:
                            pos_rule_en.append(tag_comb_en)
            final = sorted(pos_rule_en, key=lambda x : x[-1], reverse=True)
            if final :
                rule.append([tag_comb_es[0], final[0][0]])
            rules.append(rule)
    for element in rules:
        if element:
            final_rules.extend(element)
    return final_rules


#EXTRACTION WITH CRITERIA 1(don't take the combs that are too uncommon in the freqs of their own ngram (and language, ofc))


def rules_tat_crone (ngram):
    """Two changes from 0 to 1 or viceversa if english or spanish."""
    result = []
    final_result = []
    final_rules = []
    for rule in get_tat_nocr_rules(ngram):
        fdistmc = freq_dist_tags(ngram, get_tat_es_tagged()).most_common()
        for element in fdistmc:
            if rule[0] == element[0]:
                pos = fdistmc.index(element)
                num = round(((pos/len(fdistmc)) * 100),2)
                result.append([rule[0], num])
    sorted_result = sorted(result, key=lambda x: x[-1])
    for element in sorted_result:
        if element[-1] != 0:
            if element[-1] < 48:
                final_result.append(element)
#    return final_result
    for element in final_result:
        for rule in get_tat_nocr_rules(ngram):
            if element[0] == rule[0]:
                final_rules.append(rule)
    return final_rules

#EXTRACTION WITH CRITERIA 2(don't make rules if one part of the rule is very common and the other is too uncommon in their own corpora)

"""
Same functions as eu: check_eu_dist(), and filter_disparity().
"""

#EXTRACTION WITH BOTH CRITERIA (as long as it is the same set, it is a rule)
    
"""filter_disparity with get_check_eu_both_rules"""


"""STRICTS LIMITS"""

#rules_eu_crone (ngram), check_eu_dist, filter_disparity

def organise_strict_limit ():
    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    for i in range(2, 5):
        o = rules_eu_crone(i)
        n1 = 'eu_strict_rules_crone'
        filename = n1+str(i)+'.pkl'
        file = open(filename, 'wb')
        dump((o), file, -1)
        file.close()
        
        
# =============================================================================
# DUMP PICKLE
# =============================================================================
#os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    
"""Make nocr rules."""
#eu_nocr_rules5 = open('eu_nocr_rules5.pkl', 'wb')
#dump(rules_eu_nocr (5), eu_nocr_rules5, -1)
#eu_nocr_rules5.close()

#tat_nocr_rules5 = open('tat_nocr_rules5.pkl', 'wb')
#dump(rules_tat_nocr (5), tat_nocr_rules5, -1)
#tat_nocr_rules5.close()
    
"""Make rules by criteria 1."""
#eu_crone5 = open('eu_crone5.pkl', 'wb')
#dump(rules_eu_crone(5), eu_crone5, -1)
#eu_crone5.close()

#tat_crone5 = open('tat_crone5.pkl', 'wb')
#dump(rules_tat_crone(5), tat_crone5, -1)
#tat_crone5.close()

"""Check the distribution for criteria 2."""
#eu_dist5 = open('eu_dist5.pkl', 'wb')
#dump((check_eu_dist(get_eu_nocr_rules(5), get_fusioned_es(), get_fusioned_tr())), eu_dist5, -1)
#eu_dist5.close()

#tat_dist5 = open('tat_dist5.pkl', 'wb')
#dump((check_eu_dist(get_tat_nocr_rules(5), get_tat_es_tagged(), get_tat_tr_tagged())), tat_dist5, -1)
#tat_dist5.close()

"""Check the dist for both crit at the same time."""
#eu_both_dist5 = open('eu_both_dist5.pkl', 'wb')
#dump((check_eu_dist(get_eu_crone_rules(5), get_fusioned_es(), get_fusioned_tr())), eu_both_dist5, -1)
#eu_both_dist5.close()

#tat_both_dist5 = open('tat_both_dist5.pkl', 'wb')
#dump((check_eu_dist(get_tat_crone_rules(5), get_tat_es_tagged(), get_tat_tr_tagged())), tat_both_dist5, -1)
#tat_both_dist5.close()

# =============================================================================
# CALLING PICKLE FUNCTIONS
# =============================================================================

def get_eu_nocr_rules (ngram):
    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    n1 = "eu_nocr_rules"
    n2 = '{}'.format(ngram)
    n3 = ".pkl"
    inp = open(n1+n2+n3, 'rb')
    os.chdir('/home/alex/Desktop/To most freq POS combs')
    return load(inp)

def get_tat_nocr_rules (ngram):
    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    n1 = "tat_nocr_rules"
    n2 = '{}'.format(ngram)
    n3 = ".pkl"
    inp = open(n1+n2+n3, 'rb')
    os.chdir('/home/alex/Desktop/To most freq POS combs')
    return load(inp)

def get_eu_crone_rules (ngram):
    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    n1 = "eu_crone"
    n2 = '{}'.format(ngram)
    n3 = ".pkl"
    inp = open(n1+n2+n3, 'rb')
    os.chdir('/home/alex/Desktop/To most freq POS combs')
    return load(inp)

def get_tat_crone_rules (ngram):
    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    n1 = "tat_crone"
    n2 = '{}'.format(ngram)
    n3 = ".pkl"
    inp = open(n1+n2+n3, 'rb')
    os.chdir('/home/alex/Desktop/To most freq POS combs')
    return load(inp)

def get_check_eu_crtwo_rules (ngram):
    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    n1 = "eu_dist"
    n2 = '{}'.format(ngram)
    n3 = ".pkl"
    inp = open(n1+n2+n3, 'rb')
    os.chdir('/home/alex/Desktop/To most freq POS combs')
    return load(inp)

def get_check_tat_crtwo_rules (ngram):
    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    n1 = "tat_dist"
    n2 = '{}'.format(ngram)
    n3 = ".pkl"
    inp = open(n1+n2+n3, 'rb')
    os.chdir('/home/alex/Desktop/To most freq POS combs')
    return load(inp)

def get_check_eu_both_rules (ngram):
    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    n1 = "eu_both_dist"
    n2 = '{}'.format(ngram)
    n3 = ".pkl"
    inp = open(n1+n2+n3, 'rb')
    os.chdir('/home/alex/Desktop/To most freq POS combs')
    return load(inp)

def get_check_tat_both_rules (ngram):
    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    n1 = "tat_both_dist"
    n2 = '{}'.format(ngram)
    n3 = ".pkl"
    inp = open(n1+n2+n3, 'rb')
    os.chdir('/home/alex/Desktop/To most freq POS combs')
    return load(inp)


# =============================================================================
# FIX THE FUCKUP WITH NOCR RULE
# =============================================================================

def rename_pickles ():
    """
    path = '/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES/'
    """
#    file_ex = 'eu_both_dist2.pkl'
    os.chdir(path)
    for file in os.listdir():
        os.rename(file, 'old_'+file)
    
def rules_eu_nocr (n_for_range):
    es = get_fusioned_es()
    en = get_fusioned_tr()
    rules = []
    final_rules = []
    """len(OLD_rules_eu_nocr() with ngram2 is 180)"""
        
    for ngram in range(n_for_range, n_for_range+1):#6):
        fdistmc_es = freq_dist_tags(ngram, es).most_common()
        fdistmc_en = freq_dist_tags(ngram, en).most_common()
        for tag_comb_es in fdistmc_es:
            rule = []
            possible = sorted(permutations(tag_comb_es[0], ngram))
            pos_rule_en = []
            for element in possible:
                for tag_comb_en in fdistmc_en:
#                    if tag_comb_en[0] != tag_comb_es[0]:
                    if element == tag_comb_en[0]:
                        pos_rule_en.append(tag_comb_en)
            final = sorted(pos_rule_en, key=lambda x : x[-1], reverse=True)
#            return tag_comb_es
#            return final
            if final :
                rule.append([tag_comb_es[0], final[0][0]])
            rules.append(rule)
    for element in rules:
        if element:
            final_rules.extend(element)
    final_rules2 = []
    for rule in final_rules:
        if rule[0] != rule[1]:
            final_rules2.append(rule)
    return final_rules2
#    return final_rules

def rules_tat_nocr (n_for_range):
#    os.chdir('/home/alex/Desktop/To most freq POS combs/')
    es = get_tat_es_tagged()
    en = get_tat_tr_tagged()
#    os.chdir('/home/alex/Desktop/To most freq POS combs/ANALYSIS RULES')
    rules = []
    final_rules = []
    """len(OLD_rules_tat_nocr() with ngram2 is 180)"""
    
    for ngram in range(n_for_range, n_for_range+1):#6):
        fdistmc_es = freq_dist_tags(ngram, es).most_common()
        fdistmc_en = freq_dist_tags(ngram, en).most_common()
        for tag_comb_es in fdistmc_es:
            rule = []
            possible = sorted(permutations(tag_comb_es[0], ngram))
            pos_rule_en = []
            for element in possible:
                for tag_comb_en in fdistmc_en:
#                    if tag_comb_en[0] != tag_comb_es[0]:
                    if element == tag_comb_en[0]:
                        pos_rule_en.append(tag_comb_en)
            final = sorted(pos_rule_en, key=lambda x : x[-1], reverse=True)
#            return tag_comb_es
#            return final
            if final :
                rule.append([tag_comb_es[0], final[0][0]])
            rules.append(rule)
    for element in rules:
        if element:
            final_rules.extend(element)
    final_rules2 = []
    for rule in final_rules:
        if rule[0] != rule[1]:
            final_rules2.append(rule)
    return final_rules2
#    return final_rules

# =============================================================================
# DATA FROM RULES
# =============================================================================

def print_eu_data_rules ():
    x = PrettyTable()
    x.field_names = ['Extraction', '2', '3', '4', '5', 'TOTAL']
    total_nocr = []
    result_nocr = []
    result_nocr.append('NoCr')
    for i in range(2, 6):
        result_nocr.append(len(get_eu_nocr_rules(i)))
        total_nocr.append(len(get_eu_nocr_rules(i)))
    result_nocr.append(sum(total_nocr))
    
    total_crone = []
    result_crone = []
    result_crone.append('Cr1')
    for i in range(2, 6):
        result_crone.append(len(get_eu_crone_rules(i)))
        total_crone.append(len(get_eu_crone_rules(i)))
    result_crone.append(sum(total_crone))
    
    total_crtwo = []
    result_crtwo = []
    result_crtwo.append('Cr2')
    for i in range(2, 6):
        result_crtwo.append(len(filter_disparity(i, get_check_eu_crtwo_rules)))
        total_crtwo.append(len(filter_disparity(i, get_check_eu_crtwo_rules)))
    result_crtwo.append(sum(total_crtwo))
    
    total_both = []
    result_both = []
    result_both.append('Both')
    for i in range(2, 6):
        result_both.append(len(filter_disparity(i, get_check_eu_both_rules)))
        total_both.append(len(filter_disparity(i, get_check_eu_both_rules)))
    result_both.append(sum(total_both))
    
    x.add_row(result_nocr)
    x.add_row(result_crone)
    x.add_row(result_crtwo)
    x.add_row(result_both)
    print(x)

def print_tat_data_rules ():
    x = PrettyTable()
    x.field_names = ['Extraction', '2', '3', '4', '5', 'TOTAL']
    total_nocr = []
    result_nocr = []
    result_nocr.append('NoCr')
    for i in range(2, 6):
        result_nocr.append(len(get_tat_nocr_rules(i)))
        total_nocr.append(len(get_tat_nocr_rules(i)))
    result_nocr.append(sum(total_nocr))
    
    total_crone = []
    result_crone = []
    result_crone.append('Cr1')
    for i in range(2, 6):
        result_crone.append(len(get_tat_crone_rules(i)))
        total_crone.append(len(get_tat_crone_rules(i)))
    result_crone.append(sum(total_crone))
    
    total_crtwo = []
    result_crtwo = []
    result_crtwo.append('Cr2')
    for i in range(2, 6):
        result_crtwo.append(len(filter_disparity(i, get_check_tat_crtwo_rules)))
        total_crtwo.append(len(filter_disparity(i, get_check_tat_crtwo_rules)))
    result_crtwo.append(sum(total_crtwo))
    
    total_both = []
    result_both = []
    result_both.append('Both')
    for i in range(2, 6):
        result_both.append(len(filter_disparity(i, get_check_tat_both_rules)))
        total_both.append(len(filter_disparity(i, get_check_tat_both_rules)))
    result_both.append(sum(total_both))
    
    x.add_row(result_nocr)
    x.add_row(result_crone)
    x.add_row(result_crtwo)
    x.add_row(result_both)
    print(x)

# =============================================================================
# DATASETS (TEST) - JEREMY    
# =============================================================================

#example_file = '/home/alex/Desktop/To most freq POS combs/datasets_my_copy/es/tagged/test/pos.txt'
#ex = open(example_file, 'r').readlines()

def tokenize_datasets (file):
    new_file = []
    for line in file:
        sent = []
        proto = (line.split('\n')[0]).split(' ')
        for pair in proto:
            if pair:
                if len(pair.split('/')) == 2:
                    word, tag = pair.split('/')
                    sent.append(tuple([word, tag]))
        new_file.append(sent)  
    return new_file

def org_tokenizing ():
    """Automatize process of copying?"""
    
    root = '/home/alex/Desktop/To most freq POS combs/'
    path_copy = root + 'datasets_my_copy/'
    path_pickles = root + 'TOKENIZED DATASETS/'
#    file_names = []
#    result = []
    tokenized = []
    
    for folder in os.listdir(path_copy):
        if folder != 'README.txt':
            path_lang = path_copy+folder
            for sub_folder in os.listdir(path_lang):
                path_version = path_lang+'/'+sub_folder
                for sub_folder2 in os.listdir(path_version):
                    path_set = path_version+'/'+sub_folder2
                    new_path_set = path_set.replace(path_copy, '')
                    try: os.makedirs(path_pickles + new_path_set)
                    except: None

                    """FOR DELETING"""
#                    os.chdir(path_pickles)
#                    os.removedirs(new_path_set)
                    
                    for sub_folder3 in os.listdir(path_set):
                        path_file = path_set+'/'+sub_folder3
                        
                        """TOKENIZATION"""
                        if 'tagged' in path_file:
#                            if '/en/' in path_file:
                            open_file = open(path_file, 'r').readlines()
                            tokenized.append(tokenize_datasets(open_file))
                            
#                        file_names.append(path_file)
#                        result.append(open(path_file, 'r').readlines())
#    return file_names, result
    return tokenized

def pickle_dataset ():
    root = '/home/alex/Desktop/To most freq POS combs/'
    path_copy = root + 'datasets_my_copy/'
    path_pickles = root + 'TOKENIZED DATASETS/'
#    file_names = []
#    result = []
    tokenized = []
    
    for folder in os.listdir(path_copy):
        if folder != 'README.txt':
            path_lang = path_copy+folder
            for sub_folder in os.listdir(path_lang):
                path_version = path_lang+'/'+sub_folder
                for sub_folder2 in os.listdir(path_version):
                    path_set = path_version+'/'+sub_folder2
                    new_path_set = path_set.replace(path_copy, '')
                    if 'tagged' in path_set:
                        try: os.makedirs(path_pickles + new_path_set)
                        except: None

                    """FOR DELETING"""
#                    os.chdir(path_pickles)
#                    os.removedirs(new_path_set)
                    
                    for sub_folder3 in os.listdir(path_set):
                        path_file = path_set+'/'+sub_folder3
                        
                        """TOKENIZATION"""
                        if 'tagged' in path_file:
#                            if '/en/' in path_file:
                            open_file = open(path_file, 'r').readlines()
                            
                            npath = path_pickles+path_set.replace(path_copy, '')
                            os.chdir(npath)
                            
                            pickled = open(sub_folder3[:-4]+'.pkl', 'wb')
                            dump((tokenize_datasets(open_file)), pickled, -1)
                            pickled.close()
                            
#                            tokenized.append(tokenize_datasets(open_file))
                            
#                        file_names.append(path_file)
#                        result.append(open(path_file, 'r').readlines())
#    return file_names, result
#    return tokenized

def jer_tagged (en_ca_es):
    result = []
    for file in get_datasets_tagged(en_ca_es):
        for sent in file:
            result.append(sent)
    return result

def get_jer_tagset (en_ca_es):
    result = []
    for sent in jer_tagged(en_ca_es):
        for (word, tag) in sent:
            result.append(tag)
    return sorted(set(result))

def compare_tagsets (tagged_sents):
    result = []
    for sentence in tagged_sents:
        for (word, tag) in sentence:
            result.append(tag)
    return sorted(set(result))

def translation_to_datasets_tagset (my_tag):
    if my_tag in ['ADJ']:
        return 'ADJ'
    elif my_tag in ['ADP']:
        return 'ADP'
    elif my_tag in ['ADV']:
        return 'ADV'
    elif my_tag in ['AUX']: #or perhaps PRT?
        return 'VERB'
    elif my_tag in ['CONJ']:
        return 'CONJ'
    elif my_tag in ['DET']:
        return 'DET'
    elif my_tag in ['INTJ']: 
        return 'NOUN'
    elif my_tag in ['NOUN']:
        return 'NOUN'
    elif my_tag in ['NUM']:
        return 'NUM'
    elif my_tag in ['PART']:
        return 'X'
    elif my_tag in ['PRON']:
        return 'PRON'
    elif my_tag in ['PROPN']:
        return 'NOUN'
    elif my_tag in ['PUNCT']:
        return '.'
    elif my_tag in ['SCONJ']: 
        return 'CONJ'
    elif my_tag in ['SYM']:
        return 'X'
    elif my_tag in ['VERB']:
        return 'VERB'
    elif my_tag in ['X']: 
        return 'X'

def translate_to_datasets (ruleset):
    new_rules = []
    for rule in ruleset:
        new_rule = []
        for tup in rule:
            new_tup = []
            for tag in tup:
                new_tup.append(translation_to_datasets_tagset(tag))
            new_rule.append(tuple(new_tup))
        new_rules.append(new_rule)
    return new_rules


# =============================================================================
# REORDERING

# =============================================================================
# FOR PICKLING
# =============================================================================

#os.chdir('/home/alex/Desktop/To most freq POS combs/DATASETS MISC')

#es_datasets_tagged = open('en_datasets_tagged.pkl', 'wb')
#dump((org_tokenizing()), es_datasets_tagged, -1)
#es_datasets_tagged.close()

# =============================================================================
# PICKLING CALLING FUNCTIONS
# =============================================================================

def get_datasets_tagged (lang, dataset, sentiment):
    n1 = '/home/alex/Desktop/To most freq POS combs/TOKENIZED DATASETS/'
    n2 = '{}'.format(lang)
    n3 = '/tagged/'
    n4 = '{}'.format(dataset)
    n5 = '{}'.format(sentiment)
    n6 = '.pkl'
    inp = open(n1+n2+n3+n4+'/'+n5+n6, 'rb')
    return load(inp)

# =============================================================================
# FUNCTIONS
# =============================================================================            

def reorder_bigrams (sent, set_of_rules):
    for r in set_of_rules:
        for i in range(len(sent)):
            if sent[i][1] == r[0][0]:
                if i+1<len(sent) and sent[i+1][1] == r[0][1]:
#                    print(sent)
#                    print(r)
#                    print(sent[i]+sent[i+1])
#                    print('-'*20)
                    head = sent[:i]
                    tail = sent[i+2:]
                    body = np.array(sent[i:i+2])[list(r[1])]
                    reord_body = []
                    for element in body:
                        reord_body.append(tuple(element))
                    sent = head+reord_body+tail #here we get recursive
    return sent

def reorder_trigrams (sent, set_of_rules):
    for r in set_of_rules:
        for i in range(len(sent)):
            if sent[i][1] == r[0][0]:
                if i+1<len(sent) and sent[i+1][1] == r[0][1]:
                    if i+2<len(sent) and sent[i+2][1] == r[0][2]:
#                        print(sent)
#                        print(r)
#                        print(sent[i]+sent[i+1]+sent[i+2])
#                        print('-'*20)
                        head = sent[:i]
                        tail = sent[i+3:]
                        body = np.array(sent[i:i+3])[list(r[1])]
                        reord_body = []
                        for element in body:
                            reord_body.append(tuple(element))
                        sent = head+reord_body+tail #here we get recursive
    return sent

def reorder_tetragrams (sent, set_of_rules):
    for r in set_of_rules:
        for i in range(len(sent)):
            if sent[i][1] == r[0][0]:
                if i+1<len(sent) and sent[i+1][1] == r[0][1]:
                    if i+2<len(sent) and sent[i+2][1] == r[0][2]:
                        if i+3<len(sent) and sent[i+3][1] == r[0][3]:
#                            print(sent)
#                            print(r)
#                            print(sent[i]+sent[i+1]+sent[i+2]+sent[i+3])
#                            print('-'*20)
                            head = sent[:i]
                            tail = sent[i+4:]
                            body = np.array(sent[i:i+4])[list(r[1])]
                            reord_body = []
                            for element in body:
                                reord_body.append(tuple(element))
                            sent = head+reord_body+tail #here we get recursive
    return sent

def reorder_pentagrams (sent, set_of_rules):
    for r in set_of_rules:
        for i in range(len(sent)):
            if sent[i][1] == r[0][0]:
                if i+1<len(sent) and sent[i+1][1] == r[0][1]:
                    if i+2<len(sent) and sent[i+2][1] == r[0][2]:
                        if i+3<len(sent) and sent[i+3][1] == r[0][3]:
                            if i+4<len(sent) and sent[i+4][1] == r[0][4]:
#                                print(sent)
#                                print(r)
#                                print(sent[i]+sent[i+1]+sent[i+2]+sent[i+3]+sent[i+4])
#                                print('-'*20)
                                head = sent[:i]
                                tail = sent[i+5:]
                                body = np.array(sent[i:i+5])[list(r[1])]
                                reord_body = []
                                for element in body:
                                    reord_body.append(tuple(element))
                                sent = head+reord_body+tail #here we get recursive
#                            else:
#                                print('no rules found')
    return sent


# =============================================================================
# METHOD 1: if a lower-ngram rule creates a higher-ngram condition it is NOT reordered
# =============================================================================

def chain_reorder(corpus, ngrams_rules):
#    chain_reorder(get_datasets_tagged('es', 'test', 'neg')[:10], [penta_crego_rules, tetra_crego_rules, tri_crego_rules, bi_crego_rules])

    new_corpus = []
    
    for sent in corpus:
        five = reorder_pentagrams(sent, ngrams_rules[0])
        four = reorder_tetragrams(five, ngrams_rules[1])
        three = reorder_trigrams(four, ngrams_rules[2])
        two = reorder_bigrams(three, ngrams_rules[3])
        
        new_corpus.append(two)
    return new_corpus

# =============================================================================
# POST-PROCESSING
# =============================================================================

def delete_tags (corpus):
    raw = ''
    for sent in corpus:
        for i in range(len(sent)):
            if i+1 != len(sent):
                raw = raw+sent[i][0]+' '
            else:
                raw = raw+sent[i][0]+' \n'
    return raw

def check_differences(original, reordered):
#    original = get_datasets_tagged('es', 'test', 'neg')[:10]
    
#    reordered = chain_reorder(get_datasets_tagged('es', 'test', 'neg')[:10], eu_nocr)
    
    differences = []
    
    for i in range(len(original)):
        if original[i] != reordered[i]:
            diff = []
            diff.append([original[i], reordered[i]])
            
            differences.append(diff)
   
    return differences

# =============================================================================
# FORMAT MY RULES
# =============================================================================

ex_rule = get_eu_nocr_rules(3)[50]
ex_rule2 = get_eu_nocr_rules(3)[0]
ex_rule3 = get_eu_nocr_rules(4)[14]
ex_rule4 = get_eu_nocr_rules(4)[90]

def reformat_rule (rule):
    new_rule = []
    
    """RENAMING REPEATED TAGS""" """LEFT TO RIGHT!!"""

    new_comb1 = list(rule[0])
    counts1 = Counter(new_comb1)
    for tag, num in counts1.items():
        if num > 1:
            for i in range(1, num+1):
                new_comb1[new_comb1.index(tag)] = tag + '_' + str(i)
    new_rule.append(tuple(new_comb1))
    
    new_comb2 = list(rule[1])
    counts2 = Counter(new_comb2)
    for tag, num in counts2.items():
        if num > 1:
            """HERE FOR RIGHT TO LEFT"""
#            reverse = []
            for i in range(1, num+1):
#                reverse.append(i)
#            for i in list(reversed(reverse)):
                new_comb2[new_comb2.index(tag)] = tag + '_' + str(i)
    new_rule.append(tuple(new_comb2))
    
    """REFORMATING SLICES"""
    
    slices = []
    for tag in new_rule[1]:
        slices.append(new_rule[0].index(tag))
        
    new_rule[1] = tuple(slices)
    
    """RERENAMING TO ORIGINAL TAGS"""

    new_rule[0] = rule[0]

#    print(rule)
    return new_rule

def reformat_rules_set (rules_set):
    new_rules_set = []
    for r in rules_set:
        new_rules_set.append(reformat_rule(r))
    return new_rules_set

# =============================================================================
# PICKLING
# =============================================================================

#os.chdir('/home/alex/Desktop/To most freq POS combs/??')

#eu_nocr_reformat = open('eu_nocr_reformat.pkl', 'wb')
#dump((eu_nocr), eu_nocr_reformat, -1)
#eu_nocr_reformat.close()

def get_eu_nocr_reformat ():
    inp = open('eu_nocr_reformat.pkl', 'rb')
    return load(inp)

#eu_crone_reformat = open('eu_crone_reformat.pkl', 'wb')
#dump((eu_crone), eu_crone_reformat, -1)
#eu_crone_reformat.close()

def get_eu_crone_reformat ():
    inp = open('eu_crone_reformat.pkl', 'rb')
    return load(inp)

#pickle_file = open('tat_nocr_reformat.pkl', 'wb')
#dump((reformat_rules_set(get_tat_nocr_rules(5)), reformat_rules_set(get_tat_nocr_rules(4)), reformat_rules_set(get_tat_nocr_rules(3)), reformat_rules_set(get_tat_nocr_rules(2))), pickle_file, -1)
#pickle_file.close()

def get_tat_nocr_reformat ():
    inp = open('tat_nocr_reformat.pkl', 'rb')
    return load(inp)

#pickle_file = open('tat_crone_reformat.pkl', 'wb')
#dump((reformat_rules_set(get_tat_crone_rules(5)), reformat_rules_set(get_tat_crone_rules(4)), reformat_rules_set(get_tat_crone_rules(3)), reformat_rules_set(get_tat_crone_rules(2))), pickle_file, -1)
#pickle_file.close()

def get_tat_crone_reformat ():
    inp = open('tat_crone_reformat.pkl', 'rb')
    return load(inp)

#pickle_file = open('eu_crtwo_reformat.pkl', 'wb')
#dump((reformat_rules_set(filter_disparity(5, get_check_eu_crtwo_rules)), reformat_rules_set(filter_disparity(4, get_check_eu_crtwo_rules)), reformat_rules_set(filter_disparity(3, get_check_eu_crtwo_rules)), reformat_rules_set(filter_disparity(2, get_check_eu_crtwo_rules))), pickle_file, -1)
#pickle_file.close()

def get_eu_crtwo_reformat ():
    inp = open('eu_crtwo_reformat.pkl', 'rb')
    return load(inp)

#pickle_file = open('tat_crtwo_reformat.pkl', 'wb')
#dump((reformat_rules_set(filter_disparity(5, get_check_tat_crtwo_rules)), reformat_rules_set(filter_disparity(4, get_check_tat_crtwo_rules)), reformat_rules_set(filter_disparity(3, get_check_tat_crtwo_rules)), reformat_rules_set(filter_disparity(2, get_check_tat_crtwo_rules))), pickle_file, -1)
#pickle_file.close()

def get_tat_crtwo_reformat ():
    inp = open('tat_crtwo_reformat.pkl', 'rb')
    return load(inp)

#pickle_file = open('tat_both_reformat.pkl', 'wb')
#dump((reformat_rules_set(filter_disparity(5, get_check_tat_both_rules)), reformat_rules_set(filter_disparity(4, get_check_tat_both_rules)), reformat_rules_set(filter_disparity(3, get_check_tat_both_rules)), reformat_rules_set(filter_disparity(2, get_check_tat_both_rules))), pickle_file, -1)
#pickle_file.close()

def get_tat_both_reformat ():
    inp = open('tat_both_reformat.pkl', 'rb')
    return load(inp)

#pickle_file = open('eu_both_reformat.pkl', 'wb')
#dump((reformat_rules_set(filter_disparity(5, get_check_eu_both_rules)), reformat_rules_set(filter_disparity(4, get_check_eu_both_rules)), reformat_rules_set(filter_disparity(3, get_check_eu_both_rules)), reformat_rules_set(filter_disparity(2, get_check_eu_both_rules))), pickle_file, -1)
#pickle_file.close()

def get_eu_both_reformat ():
    inp = open('eu_both_reformat.pkl', 'rb')
    return load(inp)

# =============================================================================
# ALL SETS OF RULES
# =============================================================================
 
"""EU_NOCR"""
eu_nocr = get_eu_nocr_reformat()

"""EU_CRONE"""
eu_crone = get_eu_crone_reformat()

"""EU_CRTWO"""
eu_crtwo = get_eu_crtwo_reformat()

"""EU_BOTH"""
eu_both = get_eu_both_reformat()

"""TAT_NOCR"""
tat_nocr = get_tat_nocr_reformat()

"""TAT_CRONE"""
tat_crone = get_tat_crone_reformat()

"""TAT_CRTWO"""
tat_crtwo = get_tat_crtwo_reformat()

"""TAT_BOTH"""
tat_both = get_tat_both_reformat()

"""CREGO"""
crego = [penta_crego_rules, tetra_crego_rules, tri_crego_rules, bi_crego_rules]

"""ONE_RULE"""
one_rule = [[[('NOUN', 'NOUN', 'NOUN', 'NOUN', 'NOUN'), (1, 3, 4, 2, 0)]], [[('NOUN', 'NOUN', 'NOUN', 'NOUN'), (1, 3, 2, 0)]], [[('NOUN', 'NOUN', 'NOUN'), (1, 2, 0)]], [[('NOUN', 'ADJ'), (1, 0)]]]

"""ONE_RULE_BETTER"""
one_rule_better = [[[('.', '.', '.', '.', '.'), (1, 3, 4, 2, 0)]], [[('.', '.', '.', '.'), (1, 3, 2, 0)]], [[('.', '.', '.'), (1, 2, 0)]], [[('NOUN', 'ADJ'), (1, 0)]]]

"""RANDOM REORDERING"""
trivial = [[[('.', '.', '.', '.', '.'), (1, 3, 4, 2, 0)]], [[('.', '.', '.', '.'), (1, 3, 2, 0)]], [[('.', '.', '.'), (1, 2, 0)]], [[('.', '.'), (1, 0)]]]

def random_reordering (corpus):
    new_corpus = []
    for sent in corpus:
        shuffle(sent)
        new_corpus.append(sent)
    return new_corpus

# =============================================================================
# EVALUATE REORDERINGS
# =============================================================================

"""
check_differences(get_datasets_tagged('ca', 'test', 'neg'), chain_reorder(get_datasets_tagged('ca', 'test', 'neg'), tat_both))
"""

def reorder_corpus (file_name, reordering):
    en_to_copy = '/home/alex/Desktop/To most freq POS combs/REORDERINGS/en'
    get_datasets_tagged('es', 'train', 'strpos')
    corpus = 'get_datasets_tagged'
    langs = ['es', 'ca']
    dtset = ['train', 'test', 'dev']
    sentim = ['pos', 'neg', 'strpos', 'strneg']
    
    path = '/home/alex/Desktop/To most freq POS combs/REORDERINGS/'+file_name
    os.mkdir(path)
    for l in langs:
        os.mkdir(path+'/'+l)
        os.mkdir(path+'/'+l+'/raw/')
        for d in dtset:
            final_path = path+'/'+l+'/raw/'+d+'/'
            os.mkdir(final_path)
            for s in sentim:
                c = eval(corpus)(l, d, s)
                r = chain_reorder(c, reordering)
#                r = chain_reorder(random_reordering(c), reordering)
                raw = delete_tags(r)
                
                file_name = final_path+s+'.txt'
                with open(file_name, 'w') as f:
                    f.write(raw)
    shutil.copytree(en_to_copy, path+'/en/')

def evaluate(directory, lang, classifier, binary):
    #example_function = 'python3 test_reordering.py /home/alex/Desktop/pre-reordering/datasets/es/raw/train -l es -c cnn -b False'
    
    """EVALUATION FUNCTION"""
    path_reordering = '/home/alex/Desktop/pre-reordering/'
    os.chdir(path_reordering)
    function = 'python3 test_reordering.py '
    language = lang+'/raw/'
    lang_command = ' -l '+lang+' '
    classif = '-c '+classifier
    if binary == 'bi':
        bi = ' -b True'
    else:
        bi = ' -b False'
        
    command = function+directory+language+'train'+lang_command+classif+bi
    os.system(command)
    eval_file = path_reordering+'eval_reordering.txt'
    with open(eval_file, 'r') as f:
        macro_f1_train = f.readlines()[-1][10:]
        
    command = function+directory+language+'test'+lang_command+classif+bi
    os.system(command)
    eval_file = path_reordering+'eval_reordering.txt'
    with open(eval_file, 'r') as f:
        macro_f1_test = f.readlines()[-1][10:]
        
    command = function+directory+language+'dev'+lang_command+classif+bi
    os.system(command)
    eval_file = path_reordering+'eval_reordering.txt'
    with open(eval_file, 'r') as f:
        macro_f1_dev = f.readlines()[-1][10:]
    
    """OUTPUT INTO TABLE"""
    x = PrettyTable()
#    x.add_column('train', macro_f1_train)
#    x.add_column('test', macro_f1_test)
#    x.add_column('dev', macro_f1_dev)
    x.field_names = [lang+'_train', lang+'_test', lang+'_dev']
#    x.field_names = [lang, lang, lang]
#    x.add_row(['train', 'test', 'dev'])
    x.add_row([macro_f1_train, macro_f1_test, macro_f1_dev])
    os.chdir('/home/alex/Desktop/To most freq POS combs')    
    
#    return print(x)
    return x.get_string()
#    return x


def output_evals (reordering, directory, classifier):
    path = '/home/alex/Desktop/To most freq POS combs/EVAL_REORDERINGS/'
    file_name = reordering+'.txt'
    os.chdir(path)
    with open(path+file_name, 'a') as f:
        f.write('BINARY = yes')
        f.write('\n'*2)
        f.write('ENGLISH(constant)')
        f.write('\n')
        x = PrettyTable()
        x.field_names = ['train', 'test', 'dev']
        x.add_row([0.831, 0.727, 0.768])
        f.write(x.get_string())
        f.write('\n'*2)
        
        """
        EVALS BINARY
        """
        es_eval = evaluate(directory, 'es', classifier, 'bi')
        f.write(es_eval)
        f.write('\n')
        ca_eval = evaluate(directory, 'ca', classifier, 'bi')
        f.write(ca_eval)
        f.write('\n'*2)
        f.write('#'*20)
        f.write('\n')

        
        f.write('BINARY = no')
        f.write('\n'*2)
        f.write('ENGLISH(constant)')
        f.write('\n')
        y = PrettyTable()
        y.field_names = ['train', 'test', 'dev']
        y.add_row([0.989, 0.575, 0.575])
        f.write(y.get_string())
        f.write('\n'*2)
        
        """
        EVALS NOT BINARY
        """
        es_eval_no_bi = evaluate(directory, 'es', classifier, 'no')
        f.write(es_eval_no_bi)
        f.write('\n')
        ca_eval_no_bi = evaluate(directory, 'ca', classifier, 'no')
        f.write(ca_eval_no_bi)
        
    os.chdir('/home/alex/Desktop/To most freq POS combs/')

#reorder_corpus('crego', crego)    
#output_evals('crego', '/home/alex/Desktop/To\ most\ freq\ POS\ combs/REORDERINGS/crego/')

# =============================================================================
# NEW BETTER EVAL/REORDER/PRINTING EVEYTHING

reorderings_names = ['trivial', 'random_reordering()', 'one_rule_better', 'crego', 'tat_both']

def do_everything_new (filename, reordering, classifier):
    reorder_corpus(filename, reordering)
    path = '/home/alex/Desktop/To\ most\ freq\ POS\ combs/REORDERINGS/'
    output_evals(filename, path+filename+'/', classifier)
    quick_evals_new(filename)

def quick_evals_new (filename):
    path = '/home/alex/Desktop/To most freq POS combs/EVAL_REORDERINGS/'
    
    nums_binary = [12, 17]
    nums_not_binary = [-2, -7]
    slices = [2, 7, 10]
    
    file = path+filename+'.txt'

    evaluation = open(file, 'r').readlines()
    total_binary = []
    total_binary.append(evaluation[12].split(' ')[2])
    total_binary.append(evaluation[12].split(' ')[7])
    total_binary.append(evaluation[12].split(' ')[10])
    total_binary.append(evaluation[17].split(' ')[2])
    total_binary.append(evaluation[17].split(' ')[7])
    total_binary.append(evaluation[17].split(' ')[10])
    
    total_not_binary = []
    total_not_binary.append(evaluation[-2].split(' ')[2])
    total_not_binary.append(evaluation[-2].split(' ')[7])
    total_not_binary.append(evaluation[-2].split(' ')[10])
    total_not_binary.append(evaluation[-7].split(' ')[2])
    total_not_binary.append(evaluation[-7].split(' ')[7])
    total_not_binary.append(evaluation[-7].split(' ')[10])
    
    total_binary_int = []
    total_not_binary_int = []
    for string in total_binary:
        total_binary_int.append(float(string))
    for string in total_not_binary:
        total_not_binary_int.append(float(string))
        
    total_eval_binary = round((sum(total_binary_int)/len(total_binary_int)),3)
    total_eval_not_binary = round((sum(total_not_binary_int)/len(total_not_binary_int)),3)
    
    with open(file, 'a') as file_eval:
        file_eval.write('\n')
        file_eval.write('TOTAL_BINARY')
        file_eval.write('\n')
        file_eval.write(str(total_eval_binary))
        
        file_eval.write('\n')
        file_eval.write('TOTAL_NOT_BINARY')
        file_eval.write('\n')
        file_eval.write(str(total_eval_not_binary))


def compare_evals ():
    path = '/home/alex/Desktop/To most freq POS combs/EVAL_REORDERINGS/'
    result = []
    
    for file in os.listdir(path):
        f = open(path+file, 'r').readlines()
        evals = []
        total_binary = f[-3].split('\n')[0]
        total_not_binary = f[-1].split('\n')[0]
        
        evals.append([file.split('.txt')[0], total_binary, total_not_binary])
        result.extend(evals)
    
    return result

bi_most_common = sorted(compare_evals(), key = lambda x: x[-2], reverse = True)
non_bi_most_common = sorted(compare_evals(), key = lambda x: x[-1], reverse = True)

def print_evals ():
    x = PrettyTable()
    y = PrettyTable()
    
    x.field_names = ['Reordering', 'binary', 'non_binary']
    y.field_names = ['Reordering', 'binary', 'non_binary']
    for i in bi_most_common:
        x.add_row([i[0], i[1], i[2]])
    for i in non_bi_most_common:
        y.add_row([i[0], i[1], i[2]])
    
    print(x)
    print(y)

# =============================================================================
# LEXICON EXPERIMENTING
# =============================================================================

lexicon_path = '/home/alex/Desktop/pre-reordering/lexicons/bingliu_en_es.one-2-one.txt'

def parse_lexicon ():
    lex = open(lexicon_path, 'r').readlines()
    es_words = []
    for sent in lex:
        eng,es = sent.split('\t')
        es_words.append(es[:-1])
    return es_words

def testing_lexicon (sentiment):
    file = '/home/alex/Desktop/pre-reordering/datasets/es/raw/train/'+sentiment+'.txt'
    path = '/home/alex/Desktop/To most freq POS combs/REORDERINGS/change_not_lexicon/es/raw/train'
    opened = open(file, 'r').readlines()
    new_file = []
    
    for line in opened:
        new_sent = []
        sent = line.split('\n')[0]
        tokens = sent.split(' ')
        
        for t in tokens:
#            if t not in parse_lexicon() or t == '.':
            if t not in parse_lexicon():
                new_sent.append('a')
            else:
                new_sent.append(t)
        new_file.append(new_sent)
    
    raw = ''
    for sent in new_file:
        for i in range(len(sent)):
            if i+1 != len(sent):
                raw = raw+sent[i]+' '
            else:
                raw = raw+sent[i]+' \n'
    os.chdir(path)
#    with open(sentiment+'.txt', 'w') as arxiu:
#        arxiu.write(raw)

"""
no_lexicon
reduce_to_lexicon
'/home/alex/Desktop/To most freq POS combs/REORDERINGS/reduce_to_lexicon/es/raw/train/neg.txt'
python3 test_reordering.py datasets/es/raw/dev/ -l es -c bilstm -b True
"""
# =============================================================================
# ERROR ANALYSIS
# =============================================================================

or_clasif_es = load(open('/home/alex/Desktop/pre-reordering/orig_clasif.pkl', 'rb'))
or_clasif_ca = load(open('/home/alex/Desktop/pre-reordering/orig_clasif_ca.pkl', 'rb'))
or_clasif_es_ES_TRAINED = load(open('/home/alex/Desktop/pre-reordering/orig_clasif_es_ES_TRAINED.pkl', 'rb'))
or_clasif_ca_ES_TRAINED = load(open('/home/alex/Desktop/pre-reordering/orig_clasif_ca_ES_TRAINED.pkl', 'rb'))

crego_clasif_es = load(open('/home/alex/Desktop/pre-reordering/crego_clasif_es.pkl', 'rb'))
crego_clasif_ca = load(open('/home/alex/Desktop/pre-reordering/crego_clasif_ca.pkl', 'rb'))
crego_clasif_es_ES_TRAINED = load(open('/home/alex/Desktop/pre-reordering/crego_clasif_es_ES_TRAINED.pkl', 'rb'))
crego_clasif_ca_ES_TRAINED = load(open('/home/alex/Desktop/pre-reordering/crego_clasif_ca_ES_TRAINED.pkl', 'rb'))

one_rule_better_clasif_es = load(open('/home/alex/Desktop/pre-reordering/one_rule_better_clasif_es.pkl', 'rb'))
one_rule_better_clasif_ca = load(open('/home/alex/Desktop/pre-reordering/one_rule_better_clasif_ca.pkl', 'rb'))
one_rule_better_clasif_es_ES_TRAINED = load(open('/home/alex/Desktop/pre-reordering/one_rule_better_clasif_es_ES_TRAINED.pkl', 'rb'))
one_rule_better_clasif_ca_ES_TRAINED = load(open('/home/alex/Desktop/pre-reordering/one_rule_better_clasif_ca_ES_TRAINED.pkl', 'rb'))


tat_both_clasif_es = load(open('/home/alex/Desktop/pre-reordering/tat_both_clasif_es.pkl', 'rb'))
tat_both_clasif_ca = load(open('/home/alex/Desktop/pre-reordering/tat_both_clasif_ca.pkl', 'rb'))
tat_both_clasif_es_ES_TRAINED = load(open('/home/alex/Desktop/pre-reordering/tat_both_clasif_es_ES_TRAINED.pkl', 'rb'))
tat_both_clasif_ca_ES_TRAINED = load(open('/home/alex/Desktop/pre-reordering/tat_both_clasif_ca_ES_TRAINED.pkl', 'rb'))


es_trained_svm_clasif = load(open('/home/alex/Desktop/pre-reordering/es_train_svm_clasif.pkl', 'rb'))

es_trained_cnn_clasif = load(open('/home/alex/Desktop/pre-reordering/es_train_cnn_clasif.pkl', 'rb'))


def check_divergences (first_clasification, second_clasification):
    result = []
    for i in range(len(first_clasification)):
        if first_clasification[i][0] != second_clasification[i][0]:
            result.append([first_clasification[i], second_clasification[i]])
    return result

def check_mistakes (first_clasification, second_clasification):
    result = []
    for i in range(len(first_clasification)):
        gold_first = first_clasification[i][0][0]
        pred_first = first_clasification[i][0][1]
        gold_second = second_clasification[i][0][0]
        pred_second = second_clasification[i][0][1]
        
        if gold_first == pred_first and gold_second != pred_second:
            result.append([first_clasification[i], second_clasification[i]])
            
    return result

#check_mistakes(or_clasif_es, crego_clasif_es) == check_divergences(or_clasif_es, crego_clasif_es)

# =============================================================================
#  % POS/NEG DIFFERENCE
# =============================================================================
    
#classifications are up in the document
    
def perc_preds (classification):
    pos_preds = []
    neg_preds = []
    
    for el in classification:
        if el[0][1] == 0:
            neg_preds.append(el[0][1])
        elif el[0][1] == 1:
            pos_preds.append(el[0][1])
            
    return neg_preds, pos_preds

#print(len(open('/home/alex/Desktop/pre-reordering/datasets/ca/raw/dev/neg.txt', 'r').readlines()))

# =============================================================================
# LAST MINUTE TABLES 
# =============================================================================

def count_sents_corpora (lang, dataset):
    total = []
    sents = ['pos', 'neg', 'strpos', 'strneg']
    
    for s in sents:
        num = len(get_datasets_tagged(lang, dataset, s))
        total.append(num)
    
    return sum(total)

"""
(num_sents (num.words), av.sent, av.word)

CoStEP
    ES:4000
    EN:4000
TATOEBA
    ES:4000
    EN:4000
MULTIBOOKED
    CA:1149
OPENER
    ES:1472
    EN:1731
    
"""
def sentences (genre):
    return len(brown.sents(categories=genre))

def words (genre):
    return len(brown.words(categories=genre))

def average_sent (genre):
    return '{:.2f}'.format(round(words(genre)/sentences(genre),2))

def average_word (genre):
    result = []
    for w in brown.words(categories=genre):
        result.extend(w)
    return '{:.2f}'.format(round(len(result)/words(genre),2))

#corpora = [get_fusioned_es(), get_fusioned_tr(), get_tat_es_tagged(), get_tat_en_tagged(), multibooked, opener_es, opener_en]

#multibooked = []
#opener_es = []
#opener_en = []
#sentiments = ['pos', 'neg', 'strpos', 'strneg']
#datasets = ['train', 'test', 'dev']

def do_averages (corpus):
    sents = corpus
    words = []
    for s in corpus:
        words.extend(s)
    aver_sent = round((len(words)/len(sents)),4)
    corpus_characs = delete_tags(corpus)
    aver_word = round((len(corpus_characs)/len(words)),4)
    
    print(len(sents))
    print(len(words))
    print(aver_sent)
    print(aver_word)
    
def get_tagsets (corpus):
    result = []
    result.extend(corpus)
    words = []
    tags = []
    for s in result:
        words.extend(s)
    for w, pos in words:
        tags.append(pos)
    return tags
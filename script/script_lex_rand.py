import nltk
import os
import argparse
from random import shuffle

#LEXICON

def parse_lexicon(lexicon_path, eng=False):
    """
    Reads a bilingual sentiment lexicon
    and returns the set of either
    source or target language sentiment
    terms.
    """

    lexicon = []

    # Keep basic punctuation
    lexicon.append(".")
    for line in open(lexicon_path):
        source, target = line.split('\t')
        if eng:
          lexicon.append(source)
        else:
          lexicon.append(target[:-1])

    # It is faster to check if an element is a member of a set
    return set(lexicon)


def modify_to_lexicon (infile, outfile, language, mod_type="only_lex"):

    # get sentiment lexicon
    if language in ["EN", "en", "eng", "english"]:
        lexicon = parse_lexicon("lexicons/bingliu_en_es.one-2-one.txt", eng=True)
    else:
        lexicon = parse_lexicon("lexicons/bingliu_en_{0}.one-2-one.txt".format(language))

    # If mod_type is 'only_lex', create a version where all words that
    # are not found in the lexicon are replaced with UNK
    if mod_type == 'only_lex':

        modified_file = []

        for line in open(infile):
            tokens = nltk.word_tokenize(line)
            # If t is in lexicon, keep it, otherwise "UNK"
            modified = [t if t in lexicon else "UNK" for t in tokens]
            if len(modified) > 0 and modified[0] != ".":
                modified_file.append(" ".join(modified))


    # If mod_type is 'no_lex', create a version where all words that
    # are found in the lexicon are replaced with UNK
    elif mod_type == 'no_lex':

        modified_file = []

        for line in open(infile):
            tokens = nltk.word_tokenize(line)
            # If t is NOT in lexicon, keep it, otherwise "UNK"
            modified = [t if t not in lexicon else "UNK" for t in tokens]
            if len(modified) > 0 and modified[0] != ".":
                modified_file.append(" ".join(modified))

    # If mod_type is 'random' create a random permutation of word order
    # in each sentence
    elif mod_type == 'random':

        modified_file = []
        for line in open(infile):
            tokens = nltk.word_tokenize(line)
            shuffle(tokens)
            modified_file.append(" ".join(tokens))


    # Write to outfile
    with open(outfile, "w") as out:
        for line in modified_file:
            out.write(line + '\n')



def modify_directory(indir, outdir, language, mod_type="only_lex"):

    for file in os.listdir(indir):
        print(file)
        infile = os.path.join(indir, file)
        outfile = os.path.join(outdir, file)
        modify_to_lexicon(infile, outfile, language, mod_type)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_directory', help='dataset directory (directory with the txts we want to use to get its scrambled version, only_lex, and/or no_lex version')
    parser.add_argument('output_directory', help='new dataset directory (where to print the modified versions)')
    parser.add_argument('-l', '--language', default="en", help='language of the lexicon (en (default), es, ca)')
    parser.add_argument('-m', '--mod_type', default="only_lex", help='modification: "only_lex" (default), "no_lex", "random".')

    args = parser.parse_args()

    modify_directory(args.dataset_directory, args.output_directory,
                     args.language, args.mod_type)

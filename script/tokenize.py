import nltk

def tokenize ():
    inpath = '/home/alex/Desktop/test_training/datasets/trans_original/'
    outpath = '/home/alex/Desktop/test_training/datasets/trans_original_tok/'
    lang = ['en', 'ca', 'es']
    
    for l in lang:
        indir = inpath+l+'/raw/'
        outdir = outpath+l+'/raw/'
        
        for file in os.listdir(indir):
            infilename = indir+file
            outfilename = outdir+file
            new_file = []
            with open(infilename, 'r') as inf:
                for sentence in inf.readlines():
                    tokens = nltk.word_tokenize(sentence)
                    
                    new_sent = ' '.join(tokens)
                    new_file.append(new_sent)
            
            with open(outfilename, 'w') as outf:
                for sentence in new_file:
                    outf.write(sentence)
                    outf.write('\n')

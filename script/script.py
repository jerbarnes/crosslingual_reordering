import argparse
import os

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('trg_dataset', nargs= '?', default="datasets/original/es/raw", help='target dataset (directory that contains the test corpus, defaults to datasets/original/es/raw)')
    
    parser.add_argument('-l', '--lang', default='es', help='choose target language: es or ca (defaults to es)')
    parser.add_argument('-e', '--embedding', default='artetxe', help='embeddings (defaults to artetxe)')
    parser.add_argument('-c', '--classifier', default='bilstm', help='classifier (defaults to bilstm)')
    parser.add_argument('-b', '--binary', default=False, help='whether to use binary or 4-class (defaults to False == 4-class)')
    parser.add_argument('-t', '--train', default=False, help='whether to train or evaluate (defaults to False == evaluate). Training trains all 3 classifiers for es and ca.')
    
    args = parser.parse_args()
    for el in vars(args):
        print((str(el) + ': ' + str(vars(args)[el])))
    
    #Train classifiers
    if args.train in ['True', 'true']:
        print('training classifiers')       
        for c in ['bilstm', 'cnn', 'svm']:
            for l in ['es', 'ca']:
                for b in ['True', 'False']:
                    print('python3 artetxe_'+c+'.py -l '+l+' -b '+b)
                    os.system('python3 artetxe_'+c+'.py -l '+l+' -b '+b)
                            
    #Evaluate classifier
    else:
        os.system('python3 test_reordering.py {} -l {} -e {} -c {} -b {}'.format(args.trg_dataset, args.lang, args.embedding, args.classifier, args.binary))
        

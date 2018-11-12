import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('trg_dataset', nargs= '?', default="datasets/original/es/raw", help='target dataset (directory that contains the test corpus)')
    parser.add_argument('src_dataset', nargs= '?', default="datasets/divided/en/raw", help='source dataset (directory that contains the source corpus')
    
    parser.add_argument('-l', '--lang', default='es',
                        help='choose target language: es or ca (defaults to es)')
    parser.add_argument('-e', '--embedding', default='artetxe', help='embeddings')
    parser.add_argument('-c', '--classifier', default='bilstm', help='classifier')
    parser.add_argument('-b', '--binary', default=False, help='whether to use binary or 4-class (defaults to False == 4-class)')
    parser.add_argument('-t', '--train', default=False, help='whether to train or evaluate the classifier')
    
    args = parser.parse_args()
    print(args)   
    
    #Train classifier
    if args.train in ['True', 'true']:
        print('training classifier...')
#        os.system('python3 artetxe_{}.py -l {} -b {} -sd {} -td {}'.format(args.classifier, args.lang, args.binary, args.src_dataset, args.trg_dataset))
        print('python3 artetxe_{}.py -l {} -b {} -sd {} -td {}'.format(args.classifier, args.lang, args.binary, args.src_dataset, args.trg_dataset))
        
    #Evaluate classifier
    else:
        os.system('python3 test_reordering.py {} -l {} -e {} -c {} -b {}'.format(args.trg_dataset, args.lang, args.embedding, args.classifier, args.binary))
#        print('python3 test_reordering.py {} -l {} -e {} -c {} -b {}'.format(args.trg_dataset, args.lang, args.embedding, args.classifier, args.binary))
        
#python3 test_reordering.py datasets/es/raw/dev/ -l es -b True

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('test_dir', help='test directory')
    parser.add_argument('source_dir', help='source directory')
    
    parser.add_argument('-l', '--lang', default='es',
                        help='choose target language: es or ca (defaults to es)')
    parser.add_argument('-e', '--embedding', default='artetxe', help='embeddings')
    parser.add_argument('-c', '--classifier', default='bilstm', help='classifier')
    parser.add_argument('-b', '--binary', default=False, help='whether to use binary or 4-class (defaults to False == 4-class)')
    
    args = parser.parse_args()
    print(args)   
        
    os.system('python3 test_reordering.py {} -l {} -e {} -c {} -b {}'.format(args.test_dir, args.lang, args.embedding, args.classifier, args.binary))
    

#python3 test_reordering.py datasets/es/raw/dev/ -l es -b True
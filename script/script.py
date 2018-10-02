import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('test_dir', help='test directory')
    parser.add_argument('source_dir', help='source directory')
    
    parser.add_argument('-e', '--embedding', default='artexte', help='embeddings')
    parser.add_argument('-c', '--classifier', default='bilstm', help='classifier')
    
    args = parser.parse_args()
    print(args)   
        
    os.system('python3 test_reordering.py {} -l es -b True'.format(args.test_dir))
    

#python3 test_reordering.py datasets/es/raw/dev/ -l es -b True
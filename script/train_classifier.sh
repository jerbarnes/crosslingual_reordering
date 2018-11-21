#!/bin/bash

# Set to your embeddings
source_embeddings=embeddings/google.txt
target_embedding_dir=embeddings

for lang in es ca; do
	for binary in True False; do
		# train SVM
		python3 artetxe_svm.py -l $lang -se "$source_embeddings" -te "$target_embedding_dir"/sg-300-"$lang".txt -b "$binary" -td datasets/training/"$lang"/raw/

		# train biLSTM
		python3 artetxe_bilstm.py -l $lang -se "$source_embeddings" -te "$target_embedding_dir"/sg-300-"$lang".txt -b "$binary" -td datasets/training/"$lang"/raw/

		# train CNN
		python3 artetxe_cnn.py -l $lang -se "$source_embeddings" -te "$target_embedding_dir"/sg-300-"$lang".txt -b "$binary" -td datasets/training/"$lang"/raw/
	done;
done;

#!/usr/bin/env bash
# Author: Suzanna Sia
# Last modified: 10 February 2023

# CMV Data from:
# https://chenhaot.com/pages/changemyview.html


[[ ! -d data ]] && mkdir data 
cd data
echo "Downloading and unpacking Change My View dataset..."
wget --no-check-certificate https://chenhaot.com/data/cmv/cmv.tar.bz2  || (echo "Failed to download cmv data .. Error code $?" && exit 0)


 recursive unzip data
echo "Unzipping data.. this may take a while"
tar -xf cmv.tar.bz2 && rm cmv.tar.bz2
for file in `find . -type f -name "*bz2"`; do 
  bzip2 -d $file || (echo "Failed to unzip cmv data.." && exit 0)
done

echo "Downloading IQ2 Debates Corpus.."
wget http://www.ccis.neu.edu/home/kechenqin/paper/IQ2_corpus.zip -O IQ2_corpus.zip unzip IQ2_corpus.zip
rm IQ2_corpus.zip

echo "Downloading Glove embeddings.."
cd .. 
[[ ! -d data/embeddings ]] && mkdir data/embeddings
ZIP_FIL=data/embeddings/glove.6B.zip
UNZIP_FIL=data/embeddings/glove.6B.300d.txt
TMP_FIL=data/embeddings/glove_w2vec.txt

wget -O $ZIP_FIL http://nlp.stanford.edu/data/glove.6B.zip || (echo "Failed to download glove embeddings.. Error code $?" && exit 0)
unzip -d data/embeddings $ZIP_FIL
rm $ZIP_FIL

echo "prepping glove embeddings.. This may take a while"
python code/utils/prep_glove.py 

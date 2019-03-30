#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

src=src
tgt=tgt
lang=story.no.bpe
prep=generation.tokenized.nobpe
tmp=$prep/tmp
orig=orig
DATA_PATH='./generation'

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
mv ../$DATA_PATH $lang 

cd ..
prefix='toy'
echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $orig/$lang/$prefix.train.tags.$l > $tmp/$prefix.valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $orig/$lang/$prefix.train.tags.$l > $tmp/$prefix.train.$l
    cat  $orig/$lang/$prefix.test.tags.$l > $tmp/$prefix.test.$l
done

for L in $src $tgt; do
    for f in $prefix.train.$L $prefix.valid.$L $prefix.test.$L; do
        cp $tmp/$f $prep/$f
    done
done

#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt
BPE_TOKENS=10000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=src
tgt=tgt
lang=story
prep=generation.tokenized
tmp=$prep/tmp
orig=orig
DATA_PATH='./generation'

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
mv ../$DATA_PATH $lang 

cd ..

echo "pre-processing train data..."
prefix='toy'
for l in $src $tgt; do
    f=$prefix.train.tags.$l
    tok=$prefix.train.tags.tok.$l

    cat $orig/$lang/$f | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
#perl $CLEAN -ratio 1.5 $tmp/$prefix.train.tags.tok $src $tgt $tmp/$prefix.train.tags.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/$prefix.train.tags.tok.$l > $tmp/$prefix.train.tags.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    f=$prefix.test.tags.$l
    tok=$prefix.test.tags.tok.$l

    cat $orig/$lang/$f | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    perl $LC < $tmp/$prefix.test.tags.tok.$l > $tmp/$prefix.test.tags.$l
done

echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/$prefix.train.tags.$l > $tmp/$prefix.valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/$prefix.train.tags.$l > $tmp/$prefix.train.$l
    cat $tmp/$prefix.test.tags.$l > $tmp/$prefix.test.$l
done

TRAIN=$tmp/$prefix.train
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/$prefix.train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in $prefix.train.$L $prefix.valid.$L $prefix.test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

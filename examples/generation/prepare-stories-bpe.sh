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

orig=/home/ruil/generation/ruimodified_fairseq/fairseq/examples/language_model/wikitext-2
prep=bpe
tmp=$prep/tmp
DATA_PATH='./bpe'

mkdir -p $tmp $prep

echo "pre-processing train data..."
prefix='wiki'
train='train'
test='test'
valid='valid'


for l in $train $test $valid; do
  f=$prefix.$l.tokens
  tok=$prefix.$l.tokens

  cat $orig/$f | \
  perl $TOKENIZER -threads 8 -l 'src' > $tmp/$tok
  echo ""
  perl $LC < $tmp/$tok > $tmp/$tok.tok
done

TRAIN=$tmp/$prefix.train.tokens
BPE_CODE=$prep/code
rm -f $TRAIN
cat $tmp/$tok.tok >> $TRAIN


echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for l in $train $test $valid; do
    for f in $prefix.$l.tokens; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

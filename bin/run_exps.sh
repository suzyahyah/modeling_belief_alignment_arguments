#!/usr/bin/env bash
# Author: Suzanna Sia
# Last modified: 11 February 2023
# This code was first written in 2019 and I don't recommend this way of setting up experimental parameters. 

DATASET=cmv #cmv or IQ2
SAVEDIR=models
seed=0

mkdir -p $SAVEDIR

if [ `dnsdomainname` = clsp.jhu.edu ]; then
  CUDA=`free-gpu`
  echo "free gpu : $CUDA"
  export CUDA_VISIBLE_DEVICE=$CUDA
else
  echo "Your own way of getting GPUs"
fi

if [ "$DATASET" == "IQ2" ]; then
  bash ./bin/debates-data-prep.sh $seed
  dataset=${DATASET}_corpus
  glovematrix=data/embeddings/all/glove_matrix.debates
  vocabfn=data/embeddings/all/glove.vocab.debates
  trainfn=data/$dataset/train
  validfn=data/$dataset/valid
  testfn=data/$dataset/test
  testfn2=data/$dataset # not int use
  metric=acc
  hidden_dim=256
  latent_dim=128
  contrast_thresh=0.1
  margin_thresh=0 #balanced dataset so no need for margin_thresh
  
elif [ "$DATASET" == "cmv" ]; then
  glovematrix=data/embeddings/all/glove_matrix
  vocabfn=data/embeddings/all/glove.vocab
  trainfn=data/$DATASET/train-allthread_pairs.jsonlist #.filter
  validfn=data/$DATASET/valid-allthread_pairs.jsonlist
  testfn=data/$DATASET/test_id-allthread_pairs.jsonlist
  testfn2=data/$DATASET/test_cd-allthread_pairs.jsonlist
  metric=roc
  hidden_dim=256
  latent_dim=128
  contrast_thresh=0
  margin_thresh=0.5
fi

# triple can only take pair_task
declare -A F=(
["DATASET"]=$DATASET
["OLD_EMB"]=data/embeddings/glove_w2vec.txt
["NEW_EMB"]=$glovematrix
["VOCAB_FN"]=$vocabfn
["TRAIN_FN"]=$trainfn
["VALID_FN"]=$validfn
["TEST_FN"]=$testfn
["TEST_FN2"]=$testfn2
["SAVEDIR"]=$SAVEDIR
["TITLE_EMBED_FN"]=data/pair_task/train_title_embed.json
)

declare -A M=(
["ENCODER"]=${2:-glove} #, universal_se, glove
["FRAMEWORK"]=${3:-rnnv} #rnn # rnn #bert
["CONFIGN"]=0
["SS_RECON_LOSS"]=0 # semi-supervised reconstruction loss
["SEED"]=${5:-0}
["CUDA"]=$CUDA
["NUM_EPOCHS"]=300
["NWORDS"]=40000
["L_EPOCH"]=0
["RNNGATE"]=lstm
["HIDDEN_DIM"]=$hidden_dim
["LATENT_DIM"]=$latent_dim
["BATCH_SIZE"]=1
["N_LAYERS"]=2
["MAX_SEQ_LEN"]=100
["EVAL_METRIC"]=$metric
["BALANCED"]=1
)

declare -A Z=(
["ZSUM"]=rnn #cnn, fnn, #weighted_avg, simple_average, similarity
["TRIPLET_THRESH"]=${7:-0.01}
["CONTRAST_THRESH"]=$contrast_thresh
["MARGIN_THRESH"]=$margin_thresh
["USE_PRIOR_MU"]=False
["HYP"]=${4:-0}
["UNIVERSAL_EMBED"]=False # change this to encoder
["Z_COMBINE"]=concat
["SCALE_PZVAR"]=1
["WORD_DROPOUT"]=0.4
["UPDATE_ITR"]=10
)

PY="python code/main.py"

echo "===Model Params==="
for var in "${!M[@]}"
do
  PY+=" --${var,,} ${M[$var]}"
  echo "| $var:${M[$var]}"
done

echo "===File Names==="
for var in "${!F[@]}"
do
  PY+=" --${var,,} ${F[$var]}"
  echo "| $var:${F[$var]}"
done

echo "===ELBO tweaks==="
for var in "${!Z[@]}"
do
  PY+=" --${var,,} ${Z[$var]}"
  echo "| $var:${Z[$var]}"
done

echo $PY
eval $PY 

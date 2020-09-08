#!/usr/bin/env bash

# Exit if a command fails
set -e

# Set julia environment. Equivalent to call the scripts below with:
#   $ julia --project=../Project.toml ...
export JULIA_PROJECT=../Project.toml

########################################################################
# Configuration

# Directories
datadir=data
corpus=mboshi
corpusdir=/mnt/matylda5/iondel/workspace/2020/ConvHMM/recipe/mboshi-french-parallel-corpus
feadir=features  # should be on a disk with large storage
trainset=train   # subset of the corpus to train on
testset=dev      # subset of the corpus to decode
expdir=exp       # root directory where will be stored the models

# Model parameters
forder=1                    # filter order
pstrength=1                 # prior strength
epochs=30                   # number of training epochs
modelconf=conf/hmm_shared.yml      # HMM configuration file
precrate=2                  # precision update rate

# We extract the name of the configuration file to have distinguishable
# experiment directory name
confname=$(basename ${modelconf%.*})

# Type of lexicon to build the decoding graph:
#   * word (".../lexicon.wrd")
#   * phone/characters (".../lexicon.char")
lexicon=$datadir/$corpus/lang/lexicon.char

# Leave empty to marginalized
marginalize="--marginalize"

# Model for initializing the alignments
#initmodel=$expdir/$corpus/${confname}_K0_p${pstrength}/emissions_100.bson

########################################################################

echo "--> Preparing data for $corpus"
if [ ! -f $datadir/$corpus/.done ]; then

    julia local/$corpus/PrepareData.jl \
        $corpusdir \
        $datadir/$corpus

    date > $datadir/$corpus/.done
else
    echo "data already prepared"
fi

echo "--> Extracting features"
if [ ! -f $feadir/$corpus/.done ]; then

    julia steps/ExtractFeatures.jl \
        -j 10 -a "-q short.q@@blade" \
        $feadir/$corpus \
        $datadir/$corpus/*/wav.scp

    date > $feadir/$corpus/.done
else
    echo "features already extracted"
fi

echo "--> Creating the initial model"
modeldir=$expdir/$corpus/${confname}_K${forder}_p${pstrength}
if [ ! -f $modeldir/.done.init ]; then

    julia steps/CreateConvHMM.jl \
        -f $forder -p $pstrength \
        $datadir/$corpus/$trainset/uttids \
        $feadir/$corpus \
        $modelconf \
        $datadir/$corpus/lang/units \
        $modeldir

    date > $modeldir/.done.init
else
    echo "initial model already created"
fi

echo "--> Preparing the alignment graphs"
if [ ! -f $modeldir/aligns/.done ]; then

    julia steps/CompileAlignGraphs.jl \
        -j 10 -a "-q short.q@@blade" \
        --add-sil \
        $datadir/$corpus/$trainset/trans.wrd \
        $datadir/$corpus/lang/lexicon.wrd \
        $modeldir/hmms.bson \
        $modeldir/aligns

    date > $modeldir/aligns/.done
else
    echo "alignment graphs already prepared"
fi

echo "--> Training"
if [ ! -f $modeldir/.done.training ]; then

    # To initialize with an other model
    # -i $initmodel \

    julia steps/Train.jl \
        -j 30 -a "-q all.q@@blade" \
        -e $epochs \
        -u $precrate \
        $datadir/$corpus/$trainset/uttids \
        $feadir/$corpus \
        $modeldir/aligns \
        $modeldir

    date > $modeldir/.done.training
else
    echo "model already trained"
fi

decodedir=$modeldir/decode/$testset
if [ ! -z $marginalize ]; then
    decodedir=${decodedir}_marginalized
fi
mkdir -p $decodedir

echo "--> Preparing decoding graph"
if [ ! -f $modeldir/decode/.done.decgraph ]; then

    julia steps/CreateDecodeGraph.jl \
        --add-sil \
        $lexicon \
        $modeldir/hmms.bson \
        $modeldir/decode/decgraph.bson

    date > $modeldir/decode/.done.decgraph
else
    echo "decoding graph already created"
fi

echo "--> Decoding"
if [ ! -f $decodedir/.done ]; then

    julia steps/Decode.jl \
        -j 10 -a "-q all.q@@blade" \
        "$marginalize" \
        $modeldir/emissions_${epochs}.bson \
        $modeldir/decode/decgraph.bson \
        $datadir/$corpus/$testset/uttids \
        $feadir/$corpus \
        ${decodedir}

    date > $decodedir/.done
else
    echo "dataset $corpus/$testset already decoded"
fi


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
corpusdir=/mnt/matylda5/iondel/workspace/2020/ConvHMM/exp/mboshi-french-parallel-corpus
feadir=features  # should be on a disk with large storage
trainset=train
expdir=exp

# Model parameters
forder=0         # filter order
pstrength=1      # prior strength
epochs=30        # number of training epochs

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
modeldir=$expdir/$corpus/convhmm_K${forder}_p${pstrength}
if [ ! -f $modeldir/.done.init ]; then

    julia steps/CreateConvHMM.jl \
        -f $forder -p $pstrength \
        $datadir/$corpus/$trainset/uttids \
        $feadir/$corpus \
        conf/hmm.yml \
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
        $datadir/$corpus/$trainset/trans.wrd \
        $datadir/$corpus/lang/lexicon \
        $modeldir/hmms.bson \
        $modeldir/aligns

    date > $modeldir/aligns/.done
else
    echo "alignment graphs already prepared"
fi

echo "--> Training"
if [ ! -f $modeldir/.done.training ]; then

    julia steps/Train.jl \
        -j 30 -a "-q all.q@@blade" \
        -e $epochs \
        $datadir/$corpus/$trainset/uttids \
        $feadir/$corpus \
        $modeldir/aligns \
        $modeldir

    date > $modeldir/.done.training
else
    echo "model already trained"
fi


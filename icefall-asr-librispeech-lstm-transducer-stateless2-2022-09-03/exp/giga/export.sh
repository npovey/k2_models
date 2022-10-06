#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=""

. path.sh

iter=468000
avg=16

iter=472000
avg=18

set -ex

if true; then
  ./lstm_transducer_stateless2/export.py \
    --use-giga-branch 1 \
    --exp-dir ./lstm_transducer_stateless2/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --iter $iter \
    --avg  $avg \
    --pnnx 1
  mv -v lstm_transducer_stateless2/exp/encoder_jit_trace-pnnx-giga.pt lstm_transducer_stateless2/exp/encoder_jit_trace-iter-$iter-avg-$avg-pnnx-giga.pt
  mv -v lstm_transducer_stateless2/exp/decoder_jit_trace-pnnx-giga.pt lstm_transducer_stateless2/exp/decoder_jit_trace-iter-$iter-avg-$avg-pnnx-giga.pt
  mv -v lstm_transducer_stateless2/exp/joiner_jit_trace-pnnx-giga.pt lstm_transducer_stateless2/exp/joiner_jit_trace-iter-$iter-avg-$avg-pnnx-giga.pt
fi

if true; then
  ./lstm_transducer_stateless2/export.py \
    --use-giga-branch 1 \
    --exp-dir ./lstm_transducer_stateless2/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --iter $iter \
    --avg  $avg \
    --jit-trace 1
  mv lstm_transducer_stateless2/exp/encoder_jit_trace-giga.pt lstm_transducer_stateless2/exp/encoder_jit_trace-iter-$iter-avg-$avg-giga.pt
  mv lstm_transducer_stateless2/exp/decoder_jit_trace-giga.pt lstm_transducer_stateless2/exp/decoder_jit_trace-iter-$iter-avg-$avg-giga.pt
  mv lstm_transducer_stateless2/exp/joiner_jit_trace-giga.pt lstm_transducer_stateless2/exp/joiner_jit_trace-iter-$iter-avg-$avg-giga.pt
fi

if true; then
  ./lstm_transducer_stateless2/export.py \
    --use-giga-branch 1 \
    --exp-dir ./lstm_transducer_stateless2/exp \
    --bpe-model data/lang_bpe_500/bpe.model \
    --iter $iter \
    --avg  $avg \
    --jit-trace 0
  mv lstm_transducer_stateless2/exp/pretrained-giga.pt lstm_transducer_stateless2/exp/pretrained-iter-$iter-avg-$avg-giga.pt
fi

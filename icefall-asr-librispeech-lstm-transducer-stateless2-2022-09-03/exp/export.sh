#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=""

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --iter 468000 \
  --avg 16 \
  --jit-trace 0

./lstm_transducer_stateless2/export.py \
  --exp-dir ./lstm_transducer_stateless2/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --iter 468000 \
  --avg 16 \
  --jit-trace 1

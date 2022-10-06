#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="7"

. path.sh

# for m in greedy_search fast_beam_search modified_beam_search; do
for m in greedy_search ; do
  for iter in 468000; do
    for avg in 16; do
      ./lstm_transducer_stateless2/decode.py \
        --use-giga-branch 1 \
        --iter $iter \
        --avg $avg \
        --exp-dir lstm_transducer_stateless2/exp \
        --max-duration 600 \
        --num-encoder-layers 12 \
        --rnn-hidden-size 1024 \
        --decoding-method $m \
        --use-averaged-model True \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8 \
        --beam-size 4
    done
  done
done

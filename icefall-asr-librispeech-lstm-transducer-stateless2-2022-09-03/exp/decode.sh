#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1"

for m in greedy_search fast_beam_search modified_beam_search; do
  for epoch in 17; do
    for avg in 1 2; do
      ./lstm_transducer_stateless2/decode.py \
        --epoch $epoch \
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

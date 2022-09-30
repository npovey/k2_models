#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1"

# for epoch in 7 6 5 4 3; do
for m in greedy_search fast_beam_search modified_beam_search; do
  # for iter in 224000 222000 220000; do
  # for iter in 232000 230000 228000 226000; do
  # for iter in 256000 254000 252000 250000 248000 246000; do
  # for iter in 258000 254000 252000 250000 248000 246000; do
  # for iter in 260000; do
  # for iter in 264000 262000; do
  # for iter in 288000 286000 284000 282000 280000 278000; do
  # for iter in 288000 286000 284000 282000 280000 278000; do
  # for iter in 308000 306000 304000 302000 300000 ; do
  for iter in 474000; do
    for avg in 8 10 12 14 16 18; do
      ./lstm_transducer_stateless2/decode.py \
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

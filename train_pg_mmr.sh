#!/usr/bin/env bash

set -x

intexit() {
    # Kill all subprocesses (all processes in the current process group)
    kill -HUP -$$
}

hupexit() {
    # HUP'd (probably by intexit)
    echo
    echo "Interrupted"
    kill $(lsof -ti :"$port")
    exit
}

trap hupexit HUP
trap intexit INT


dataset_name=cnn_dm
mode=train
singles_and_pairs=singles

cuda=0
max_enc_steps=100
min_dec_steps=10
max_dec_steps=30
batch_size=128
exp_name=""

exp_suffix=""
dataset_split=train
num_iterations=10000000
port=6006
data_root_flag=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset_name=*)
      dataset_name="${1#*=}"
      ;;
    --mode=*)
      mode="${1#*=}"
      ;;
    --singles_and_pairs=*)
      singles_and_pairs="${1#*=}"
      ;;
    --cuda=*)
      cuda="${1#*=}"
      ;;
    --max_enc_steps=*)
      max_enc_steps="${1#*=}"
      ;;
    --min_dec_steps=*)
      min_dec_steps="${1#*=}"
      ;;
    --max_dec_steps=*)
      max_dec_steps="${1#*=}"
      ;;
    --batch_size=*)
      batch_size="${1#*=}"
      ;;
    --data_root_flag=*)
      data_root_flag="${1#*=}"
      ;;
    --exp_name=*)
      exp_name="${1#*=}"
      ;;
    --finetune=*)
      finetune="${1#*=}"
      ;;
    *)
        break
  esac
  shift
done

if [[ "$mode" = "all" ]]; then
    mode=train_eval_decode_tensorboard_restore
fi

if [[ "$mode" = "eval" ]]; then
    cuda=1
    dataset_split=val
    num_iterations=-1
fi

if [[ "$mode" = "tensorboard" ]]; then
    cuda=1
    dataset_split=val
    num_iterations=-1
fi

if [[ "$singles_and_pairs" = "none" ]]; then
    data_root_flag=--data_root=$HOME/data/tf_data
fi

if [[ "$cuda" = "1" ]]; then
    port=7007
fi

if [[ "$HOME" == "/home/logan" ]]; then
    cuda_phrase="CUDA_VISIBLE_DEVICES=""$cuda"
else
    cuda_phrase=""
fi

#if [[ "$mode" == *"train"* ]]; then
#    pipe=\&> $HOME/null \&
#else
#    pipe=""
#fi

echo "$dataset_name"
echo "$mode"
echo "$singles_and_pairs"
echo "$@"



if [[ "$mode" == *"tensorboard"* ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" tensorboard --logdir=logs/"$dataset_name"_pgmmr_"$singles_and_pairs"$exp_name/eval --port="$port" &
fi
if [[ "$mode" == *"eval"* ]]; then
    if [[ "$mode" == *"train"* ]]; then
        CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode=eval --dataset_name="$dataset_name" --dataset_split=val --exp_name="$exp_suffix"$exp_name --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations=-1 $data_root_flag --pg_mmr --singles_and_pairs="$singles_and_pairs" "$@" &> $HOME/null &
    else
        CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode=eval --dataset_name="$dataset_name" --dataset_split=val --exp_name="$exp_suffix"$exp_name --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations=-1 $data_root_flag --pg_mmr --singles_and_pairs="$singles_and_pairs" "$@"
    fi
fi
if [[ "$mode" == *"train"* ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode=train --dataset_name="$dataset_name" --dataset_split="$dataset_split" --exp_name="$exp_suffix"$exp_name --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations="$num_iterations"  $data_root_flag --pg_mmr --singles_and_pairs="$singles_and_pairs" "$@"
fi
if [[ "$mode" == *"restore"* ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode=train --dataset_name="$dataset_name" --dataset_split="$dataset_split" --exp_name="$exp_suffix"$exp_name --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=False --batch_size="$batch_size" --num_iterations="$num_iterations" $data_root_flag --pg_mmr --singles_and_pairs="$singles_and_pairs" --restore_best_model "$@"
fi
if [[ "$mode" == *"decode"* ]]; then
    CUDA_VISIBLE_DEVICES="$cuda" python run_summarization.py --mode=decode --dataset_name="$dataset_name" --dataset_split=test --exp_name="$exp_suffix"$exp_name --max_enc_steps="$max_enc_steps" --min_dec_steps="$min_dec_steps" --max_dec_steps="$max_dec_steps" --single_pass=True --batch_size="$batch_size" --num_iterations=-1 $data_root_flag --pg_mmr --singles_and_pairs="$singles_and_pairs" "$@"
fi
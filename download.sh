#!/bin/bash

if [[ "$2" == "train_tiny" ]] ;then
  mkdir -p $1/train_tiny
  for i in $(seq -f "%06g" 0 9)
  do
    echo "Downloading train_tiny/shard-$i.tar.gz..." 
    wget https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/zerograsp/train_tiny/shard-$i.tar.gz -O $1/train_tiny/shard-$i.tar.gz
    echo "Decompressing train/shard-$i.tar.gz..." 
    pigz -d $1/train_tiny/shard-$i.tar.gz
  done
elif [[ "$2" == "train" ]] ;then
  mkdir -p $1/train
  for i in $(seq -f "%06g" 0 9999)
  do
    echo "Downloading train/shard-$i.tar.gz..." 
    wget https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/zerograsp/train/shard-$i.tar.gz -O $1/train/shard-$i.tar.gz
    echo "Decompressing train/shard-$i.tar.gz..." 
    pigz -d $1/train/shard-$i.tar.gz
  done
elif [[ "$2" == "woven_easy" ]] ;then
  mkdir -p $1/woven_easy
  for i in $(seq -f "%06g" 0 4)
  do
    echo "Downloading train/shard-$i.tar.gz..." 
    wget https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/zerograsp/woven_easy/shard-$i.tar -O $1/woven_easy/shard-$i.tar
  done
elif [[ "$2" == "woven_normal" ]] ;then
  mkdir -p $1/woven_normal
  for i in $(seq -f "%06g" 0 4)
  do
    echo "Downloading train/shard-$i.tar.gz..." 
    wget https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/zerograsp/woven_normal/shard-$i.tar -O $1/woven_normal/shard-$i.tar
  done
elif [[ "$2" == "woven_hard" ]] ;then
  mkdir -p $1/woven_hard
  for i in $(seq -f "%06g" 0 4)
  do
    echo "Downloading train/shard-$i.tar.gz..." 
    wget https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/zerograsp/woven_hard/shard-$i.tar -O $1/woven_hard/shard-$i.tar
  done
else
  echo "Please specify the dataset name properly."
fi
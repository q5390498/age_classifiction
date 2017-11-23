#!/usr/bin/env zsh
rm train.log
python train.py |tee >> train.log
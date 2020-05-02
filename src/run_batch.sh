#!/bin/bash
python main_seq.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.01 --w1=0.8 --w2=0.2 --v="00"
python main_seq.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.01 --w1=0.2 --w2=0.8 --v="01"
python main_seq.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.02 --w1=0.8 --w2=0.2 --v="02"
python main_seq.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.02 --w1=0.2 --w2=0.8 --v="03"
python main_seq.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.04 --w1=0.8 --w2=0.2 --v="04"
python main_seq.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.04 --w1=0.2 --w2=0.8 --v="05"
python main_seq.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.0 --w1=0.5 --w2=0.5 --v="06"
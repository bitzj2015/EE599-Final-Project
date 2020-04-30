#!/bin/bash
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.5 --w1=0.5 --w2=0.0 --v="00"
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.1 --w1=0.2 --w2=0.8 --v="01"
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.1 --w1=0.8 --w2=0.2 --v="02"
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.4 --w1=0.2 --w2=0.8 --v="03"
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.4 --w1=0.8 --w2=0.2 --v="04"
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.8 --w1=0.2 --w2=0.8 --v="05"
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.8 --w1=0.8 --w2=0.2 --v="06"
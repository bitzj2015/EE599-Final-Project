#!/bin/bash
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.1 --w1=0.2 --w2=0.8
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.1 --w1=0.8 --w2=0.2
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.4 --w1=0.2 --w2=0.8
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.4 --w1=0.8 --w2=0.2
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.8 --w1=0.2 --w2=0.8
python main.py --cuda=1 --batch_size=64 --phase="train_pri" --w0=0.8 --w1=0.8 --w2=0.2
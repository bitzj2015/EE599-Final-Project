# Command to train GAP
* Please change `GEN_PATH`, `DIS_PATH`, `ADV_PATH` and `W` in `train.py`. 
* Then run command:
`sudo python main.py --cuda=0 --batch_size=64 --phase="train_gap"`

# Experimental settings
* python main_base.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.01 --w1=0.6 --w2=0.4 --v="-W164"
* python main_base.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.01 --w1=0.4 --w2=0.6 --v="-W146"
* python main_base.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.01 --w1=0.9 --w2=0.1 --v="-W191"
* python main_base.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.01 --w1=0.1 --w2=0.9 --v="-W119"
* python main_base.py --cuda=1 --batch_size=64 --phase="train_gap" --w0=0.1 --w1=0.5 --w2=0.5 --v="-W1055"


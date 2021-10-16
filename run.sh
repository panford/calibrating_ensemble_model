#!/bin/bash

script="run.sh"

python train.py --epochs 20
python test.py --return_pred_targ 'False'
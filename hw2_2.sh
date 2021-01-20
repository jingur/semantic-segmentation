#!/bin/bash
wget -O baseline_model.pth.tar https://www.dropbox.com/s/zacqavt930b2d10/baseline_model.pth.tar?dl=1
python3 test_p2.py --input_dir $1 --output_dir $2 --resume ./log/baseline_model.pth.tar
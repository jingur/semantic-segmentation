#!/bin/bash
wget -O improved_model.pth.tar https://www.dropbox.com/s/xcz4dgg5zme1xd0/improved_model.pth.tar?dl=1
python3 test_p2_best.py --input_dir $1 --output_dir $2 --resume improved_model.pth.tar 
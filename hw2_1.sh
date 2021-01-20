#RESUME='model.pth'
wget -O model.pth https://www.dropbox.com/s/bg7mpduys39e17b/model.pth?dl=1
python3 test_p1.py --input_dir $1 --output_dir $2  --resume model.pth

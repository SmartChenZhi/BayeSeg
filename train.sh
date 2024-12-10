python train.py --model UNet --output_dir logs/unet2 --device cuda:0 --batch_size 24
python train.py --model vqUNet --output_dir logs/vqunet2 --device cuda:1 --batch_size 32
python train.py --model BayeSeg --output_dir logs/model2 --device cuda:1
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg --device cuda:0
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg2 --device cuda:1


python test.py --model UNet --checkpoint_dir logs/unet2 --device cuda:1
python test.py --model vqUNet --checkpoint_dir logs/vqunet2 --device cuda:1
python test.py --model BayeSeg --checkpoint_dir logs/model --device cuda:0
python test.py --model BayeSeg --checkpoint_dir logs/model2 --device cuda:1

tensorboard --logdir=logs --port=6018
python train.py --model UNet --output_dir logs/unet2 --device cuda:0 --batch_size 24
python train.py --model vqUNet --output_dir logs/vqunet2 --device cuda:1 --batch_size 32
python train.py --model BayeSeg --output_dir logs/model2 --device cuda:1
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg --device cuda:0
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg2 --device cuda:1
python train.py --model BayeSeg --output_dir logs/model_scaletransform2 --device cuda:1


python test.py --model UNet --checkpoint_dir logs/unet3 --device cuda:1
python test.py --model vqUNet --checkpoint_dir logs/vqunet3 --device cuda:0
python test.py --model BayeSeg --checkpoint_dir logs/model --device cuda:0
python test.py --model BayeSeg --checkpoint_dir logs/model2 --device cuda:1
python test.py --model BayeSeg --checkpoint_dir logs/model_scaletransform2 --device cuda:1

tensorboard --logdir=logs/model_scaletransform2 --port=6009
tensorboard --logdir=runs/vqvae_experiment2 --port=6020
#run3 采用初始化
python train.py --model UNet --output_dir logs/unet3 --device cuda:0 --batch_size 32 --epochs 500
python train.py --model vqUNet --output_dir logs/vqunet3 --device cuda:1 --batch_size 32 --epochs 500

python train.py --model vqvae --output_dir logs/vqvae --device cuda:0 --batch_size 32 --epochs 500
python train.py --model UNet --output_dir logs/unet2 --device cuda:0 --batch_size 24
python train.py --model vqUNet --output_dir logs/vqunet4 --device cuda:1 --batch_size 32 --epochs 500
python train.py --model vqUNet --output_dir logs/vqunet5 --device cuda:0 --batch_size 24 --epochs 500
python train.py --model BayeSeg --output_dir logs/model2 --device cuda:1
python train.py --model BayeSeg --output_dir logs/model3 --device cuda:1
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg5 --device cuda:1 --bayes_loss_coef 1 #shape,appearance pretrained
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg2 --device cuda:1
python train.py --model BayeSeg --output_dir logs/model_scaletransform2 --device cuda:1


python test.py --model UNet --checkpoint_dir logs/unet3 --device cuda:1
python test.py --model vqUNet --checkpoint_dir logs/vqunet4 --device cuda:1
python test.py --model BayeSeg --checkpoint_dir logs/model3 --device cuda:0
python test.py --model BayeSeg --checkpoint_dir logs/model2 --device cuda:1
python test.py --model BayeSeg --checkpoint_dir logs/model_scaletransform2 --device cuda:1
python test.py --model vqBayeSeg --checkpoint_dir logs/vqBayeSeg5 --device cuda:0

tensorboard --logdir=logs/model3 --port=6016
tensorboard --logdir=logs/vqBayeSeg10 --port=6010
tensorboard --logdir=runs/vqvae_experiment2 --port=6020
tensorboard --logdir=logs/vqunet4 --port=6025 #去噪01归一化数据监督(4:512,5:48)
#run3 采用初始化
python train.py --model UNet --output_dir logs/unet3 --device cuda:0 --batch_size 32 --epochs 500
python train.py --model vqUNet --output_dir logs/vqunet3 --device cuda:1 --batch_size 32 --epochs 500

python train.py --model vqvae --output_dir logs/vqvae --device cuda:0 --batch_size 32 --epochs 500

#848 75.5(88.2 82.4 51.6 83.0 80.3)
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg4 --device cuda:0
python test.py --model vqBayeSeg --checkpoint_dir logs/vqBayeSeg4 --device cuda:0
# 1084 77.6(87.8 81.7 70.4 79.9 80.5 75.3)
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg5 --device cuda:0
python test.py --model vqBayeSeg --checkpoint_dir logs/vqBayeSeg5 --device cuda:0
# loss只保留方向 1662 75.6(87.6 83.0 49.6 81.8 82.7 81.1)
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg6 --device cuda:0 --weight_decay 0.0005 --epochs 2000
python test.py --model vqBayeSeg --checkpoint_dir logs/vqBayeSeg6 --device cuda:0
# loss只保留方向 1014 77.8 (87.2 84.0 58.5 80.8 83.7 82.2)
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg7 --device cuda:1 --weight_decay 0.0005
python test.py --model vqBayeSeg --checkpoint_dir logs/vqBayeSeg7 --device cuda:1

# loss重建权重：3e-4 1014 75.6 (86.8 83.9 52.1 80.9 82.0 79.1)
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg8 --device cuda:0 --weight_decay 0.0003
python test.py --model vqBayeSeg --checkpoint_dir logs/vqBayeSeg8 --device cuda:0
# loss重建权重：100 700 73.5(88.6 82.1 38.6 80.2 80.9 83.7)
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg9 --device cuda:1 --weight_decay 0.0003
python test.py --model vqBayeSeg --checkpoint_dir logs/vqBayeSeg9 --device cuda:1
# 
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg10 --device cuda:0
python test.py --model vqBayeSeg --checkpoint_dir logs/vqBayeSeg10 --device cuda:0
# 
python train.py --model vqBayeSeg --output_dir logs/vqBayeSeg11 --device cuda:1 --weight_decay 0.0003
python test.py --model vqBayeSeg --checkpoint_dir logs/vqBayeSeg11 --device cuda:1

python train.py --model udaBayeSeg --output_dir logs/udaBayeSeg --device cuda:0 --uda
python test.py --model udaBayeSeg --checkpoint_dir logs/udaBayeSeg --device cuda:0 --uda
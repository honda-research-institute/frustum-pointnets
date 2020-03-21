#/bin/bash
python train/train.py --gpu 0 --model frustum_pointnets_lite --log_dir train/log_lite --num_point 128 --max_epoch 81 \
 --batch_size 32 --decay_step 800000 --decay_rate 0.5

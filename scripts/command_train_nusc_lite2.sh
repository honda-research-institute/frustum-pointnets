#/bin/bash
python train/train_lite.py --gpu 0 --model frustum_pointnets_lite2 --log_dir train/log_lite2 --num_point 64 \
--max_epoch 81 --batch_size 32 --decay_step 800000 --decay_rate 0.5

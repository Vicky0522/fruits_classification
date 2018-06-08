python train_test.py --dataroot /mnt/tencent/vicky/Dataset/fruits-360_dataset/fruits_360 \
                     --name fruits_classify_alexnet \
                     --model classify \
                     --which_model_netC alexnet --init_type kaiming \
                     --lr 0.001 --lr_policy step --lr_decay_iters 14 \
                     --dataset classify --batchSize 128 --loadSize 224 --fineSize 224 \
                     --norm batch --save_epoch_freq 1 \
                     --gpu_ids 1 \
                     --display_port 8101 \
                     # --continue_train --which_epoch latest --epoch_count 4
                     # --continue_train # --which_epoch latest --epoch_count 17

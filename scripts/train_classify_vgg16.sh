python train.py --dataroot /mnt/tencent/vicky/Dataset/fruits-360_dataset/fruits_360 \
                --name fruits_classify_resnet50 \
                --model classify \
                --which_model_netC resnet_50 --init_type kaiming \
                --dataset classify --batchSize 128 --loadSize 120 --fineSize 100 \
                --norm batch \
                --gpu_ids 1 \
                --display_port 8101
                # --continue_train # --which_epoch latest --epoch_count 17

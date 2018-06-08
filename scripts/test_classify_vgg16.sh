python test.py --dataroot /mnt/tencent/vicky/Dataset/fruits-360_dataset/fruits_360 \
               --name fruits_classify_vgg16 \
               --model classify \
               --which_model_netC vgg_16 \
               --dataset_mode classify --loadSize 224 --fineSize 224 \
               --norm batch \
               --which_epoch 4
               

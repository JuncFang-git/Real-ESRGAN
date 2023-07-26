# # prepare datasets
# ## get meta_info_xxx_sub_pair.txt
# python ../scripts/generate_meta_info_pairdata.py \
# --input /home/junkai/project/230522_idphoto_sr/dataset/div2k/DIV2K_train_HR \
# /data2/junkai/project/230522_idphoto_sr/dataset/div2k/DIV2K_train_LR_bicubic/X2 \
# --meta_info /data2/junkai/project/230522_idphoto_sr/dataset/div2k/meta_info/meta_info_DIV2K_sub_pair.txt


# train
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
../realesrgan/train.py \
-opt ../options/finetune_realesrgan_x2plus_my.yml \
--launcher pytorch \
--auto_resume

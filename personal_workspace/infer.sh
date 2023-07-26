# Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

# A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

#   -h                   show this help
#   -i --input           Input image or folder. Default: inputs
#   -o --output          Output folder. Default: results
#   -n --model_name      Model name. Default: RealESRGAN_x4plus
#   -s, --outscale       The final upsampling scale of the image. Default: 4
#   --suffix             Suffix of the restored image. Default: out
#   -t, --tile           Tile size, 0 for no tile during testing. Default: 0
#   --face_enhance       Whether to use GFPGAN to enhance face. Default: False
#   --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
#   --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto


cd .. && \
CUDA_VISIBLE_DEVICES=3 python inference_realesrgan.py \
-i /home/junkai/project/230522_idphoto_sr/Real-ESRGAN/personal_workspace/test_img \
-o /home/junkai/project/230522_idphoto_sr/Real-ESRGAN/personal_workspace/test_meiturestore_latest \
-n RealESRGAN_x2plus \
--model_path /home/junkai/project/230522_idphoto_sr/Real-ESRGAN/experiments/finetune_RealESRGANx2plus_4metitu_restore_paired2000/models/net_g_latest.pth \
--outscale 2

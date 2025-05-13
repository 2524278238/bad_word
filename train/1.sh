export LD_LIBRARY_PATH=/public/home/xiangyuduan/anaconda3/envs/rstar/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="0"
python train.py
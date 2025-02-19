# other utility libraries
apt-get update
apt-get install -y libgl1 libglib2.0-0 libx11-6 openssh-server tmux vim 

# pointnet
export PATH=/usr/local/cuda/bin:$PATH
pip install ../SaliencyI2PLoc/extentions/pointnet2_ops_lib/.

cd ../../../scripts

# python libraries
pip install -r ../requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# GPU kNN for network effects
pip install ../SaliencyI2PLoc/extentions/KNN_CUDA-0.2-py3-none-any.whl

# for baseline 
cp ../pretrained_models/resnet18-5c106cde.pth /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
cp ../pretrained_models/resnet18-f37072fd.pth /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
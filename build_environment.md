git submodule update --init --recursive
```bash
pip install torch
```
```bash
pip install tensorboard omegaconf h5py pyquaternion open3d FastGeodis
```

install pytorch3d (from source is recommand)
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
be patient

install torch_kdtree (from source is recommand)
```bash
pip install "git+https://github.com/thomgrand/torch_kdtree.git"
```
be patient

install detectron2
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
install torchscatter
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```
install spconv
build from source is recommand, the prebuild binary file usually don`t work. Anyway,I never use prebuild binary run anything successfully. If you can run it using prebuild binary, you are lucky guy.
```python
#after compile cumm, you need to update the requirements in spconv, 
# modify the file setup.py in spconv
if cuda_ver:
    cuda_ver_str = cuda_ver.replace(".", "") # 10.2 to 102

    RELEASE_NAME += "-cu{}".format(cuda_ver_str)
    deps = ["cumm-cu{}>=0.7.11, <0.8.0".format(cuda_ver_str)]#to 0.8.3
else:
    deps = ["cumm>=0.7.11, <0.8.0"] #to 0.8.3

```
install build-essential, install CUDA
git clone https://github.com/FindDefinition/cumm, cd ./cumm, pip install -e .
git clone https://github.com/traveller59/spconv, cd ./spconv, pip install -e .
in python, import spconv and wait for build finish.

```bash

pip install pytorch_lightning mmcv
```
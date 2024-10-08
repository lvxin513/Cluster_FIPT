# README

### Consistent Inverse Rendering of Indoor Scenes via Semantic-Aware Hierarchical Clustering

This repository includes the source code and datasets for the paper titled *"Consistent Inverse Rendering of Indoor Scenes via
Semantic-Aware Hierarchical Clustering."* We offer two synthetic scenes (kitchen and living room) and one real scene (classroom). Using the provided code and data, readers can replicate the results presented in the paper.

## Setup

Set up the environment via:

```
conda create --name ir python=3.8 pip
conda activate ir
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch # tested with tinycudann-1.7
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

### Dataset

The dataset is accessible at [Baidu Cloud](https://pan.baidu.com/s/1wSIU3ke_TbMk_YsB_BdWnQ) using the password "xkao". It includes two synthetic scenes and one real scene. For further information about dataset, please refer to the following repository: https://github.com/lwwu2/fipt.

### Train the model

**1.Initialize shadings:**

```python
# --scene: the path of train dataset; --output: the location where you store your output; 
# --dataset:synthetic or real
python bake_shading.py --scene /proj/users/xlv/lvxin/fipt/data/kitchen --output outputs/kitchen --dataset synthetic
```

**2.Optimize BRDF and emission mask:**

```python
# --dataset{default:'synthetic' or 'real', '../data/indoor_synthetic/kitchen':the path of train dataset,'outputs/kitchen':the location where you store your output on initialize shading}
python train.py --experiment_name kitchen --device 0 --max_epochs 3 --dataset synthetic /proj/users/xlv/lvxin/fipt/data/kitchen outputs/kitchen --voxel_path outputs/kitchen/vslf.npz
```

**3.Extract emitters:**

```python
python extract_emitter.py --scene /proj/users/xlv/lvxin/fipt/data/kitchen --output outputs/kitchen --dataset synthetic --ckpt checkpoints/kitchen/cluster_part.ckpt
```

**4.Shading refinement:**

```python
# --ft : the number of finetune
python refine_shading.py --scene /proj/users/xlv/lvxin/fipt/data/kitchen --output outputs/kitchen --dataset synthetic --ckpt checkpoints/kitchen/cluster_part.ckpt --ft 1
```

## Inference 

**Install conda env to Jupyter**

```
conda activate ir
python -m ipykernel install --user --name=ir
```

You can check the 'demo' directory, which contains Jupyter notebook files for both visualization and quantitative results. Besides, the application of relighting and object inserting can be find.

![image-20240826152732026](Result.png)

 
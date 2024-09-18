# data folder
DATASET_ROOT='data/indoor_synthetic/'
DATASET='synthetic'
# scene name
SCENE='kitchen'
# whether has part segmentation
HAS_PART=0

# shading initialization
# python bake_shading.py --scene /proj/users/xlv/lvxin/fipt/data/kitchen --output outputs/kitchen --dataset synthetic
# python bake_shading.py --scene /proj/users/xlv/lvxin/fipt/data/classroom --output outputs/classroom --dataset real
python bake_shading.py --scene $DATASET_ROOT$SCENE\
                       --output 'outputs/'$SCENE --dataset $DATASET

# BRDF-emission estimation
# python train.py --experiment_name kitchen --device 0 --max_epochs 1 --dataset synthetic /proj/users/xlv/lvxin/fipt/data/kitchen outputs/kitchen --voxel_path outputs/kitchen/vslf.npz
# python train.py --experiment_name classroom --device 0 --max_epochs 1 --dataset real /proj/users/xlv/lvxin/fipt/data/classroom outputs/classroom --voxel_path outputs/classroom/vslf.npz
python train.py --experiment_name $SCENE --device 0 --max_epochs 2\
        --dataset $DATASET $DATASET_ROOT$SCENE 'outputs/'$SCENE\
        --voxel_path 'outputs/'$SCENE'/vslf.npz'\
        --has_part $HAS_PART
# python extract_emitter.py --scene /proj/users/xlv/lvxin/fipt/data/kitchen --output outputs/kitchen --dataset synthetic --ckpt checkpoints/kitchen/cluster_part.ckpt
# extract emitters
python extract_emitter.py --scene $DATASET_ROOT$SCENE\
        --output 'outputs/'$SCENE --dataset $DATASET\
        --ckpt 'checkpoints/'$SCENE'/last.ckpt'

# python refine_shading.py --scene /proj/users/xlv/lvxin/fipt/data/kitchen --output outputs/kitchen --dataset synthetic --ckpt checkpoints/kitchen/cluster_part.ckpt --ft 1
# refine shadings
python refine_shading.py --scene $DATASET_ROOT$SCENE\
        --output 'outputs/'$SCENE --dataset $DATASET\
        --ckpt 'checkpoints/'$SCENE'/last.ckpt' --ft 1 


# alternative optimization
python train.py --experiment_name $SCENE'1' --device 0 --max_epochs 2\
        --dataset $DATASET $DATASET_ROOT$SCENE 'outputs/'$SCENE'1'\
        --voxel_path 'outputs/'$SCENE'/vslf.npz'\
        --has_part $HAS_PART

python refine_shading.py --scene $DATASET_ROOT$SCENE\
        --output 'outputs/'$SCENE --dataset $DATASET\
        --ckpt 'checkpoints/'$SCENE'1/last.ckpt' --ft 2

python train.py --experiment_name $SCENE'2' --device 0 --max_epochs 2\
        --dataset $DATASET $DATASET_ROOT$SCENE 'outputs/'$SCENE'2'\
        --voxel_path 'outputs/'$SCENE'/vslf.npz'\
        --has_part $HAS_PART 
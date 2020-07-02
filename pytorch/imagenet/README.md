<h1 align="center">
Madam optimiser
</h1>

## ImageNet training in PyTorch

This code is forked from [Pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
- Install NVIDIA DALI

## Training
Training with SGD
```bash
python main.py -a resnet50 --SGD --lr=0.1 --dali_cpu --workers=8 DATASET_DIR
```

Training with Adam
```bash
python main.py -a resnet50 --adam --lr=0.001 --dali_cpu --workers=8 DATASET_DIR
```

Training with Madam
```bash
python main.py -a resnet50 --init_bias --ori_madam --lr=0.01 --scale=3 --dali_cpu --workers=8 DATASET_DIR
```

Training with Int Madam
```bash
python main.py -a resnet50 --init_bias --int_madam --lr=0.001 --scale=3 --lr_factor=16 --num_levels=2048 --decay_factor=0.25 --dali_cpu --workers=8 DATASET_DIR
```

# Train CIFAR10 and CIFAR100 with Madam

The repository is forked from [PyTorch Examples](https://github.com/pytorch/examples).

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```bash
python main.py --SGD --lr=0.1 #running with SGD on cifar10

python main.py --adam --lr=0.001 #running with Adam on cifar10

python main.py --madam --lr=0.01 --scale=3 --modified_net #running with Madam on cifar10

python main.py --int_madam --lr=0.001 --scale=3 --modified_net --lr_factor=16  --divider=4 --num_levels=4096 #running with Int Madam on cifar10

python main.py --SGD --lr=0.1 --cifar100 #running with SGD on cifar100

python main.py --adam --lr=0.001 --cifar100 #running with Adam on cifar100

python main.py --madam --lr=0.01 --scale=1 --modified_net --cifar100 #running with Madam on cifar100

python main.py --int_madam --lr=0.001 --scale=1 --modified_net --lr_factor=16  --divider=4 --num_levels=4096 --cifar100 #running with Int Madam on cifar100
```


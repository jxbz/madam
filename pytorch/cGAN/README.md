<h1 align="center">
cGAN
</h1>

## CIFAR-10 class conditional GAN

The following Python packages are required: numpy, torch, torchvision, tqdm.

An example job is
```
python main.py --seed 0 --optim madam --initial_lr 0.01
```
See inside `batch.sh` for the commands run in the paper.

## Changes compared to the [Fromage repository](https://github.com/jxbz/fromage)

- Initialise biases using `init.normal_(m.bias.data, mean=0.0, std=0.01)`
instead of `init.constant_(m.bias.data, 0.0)`
- Initialise self-attention gamma using `self.gamma = nn.Parameter(torch.ones(1))`
instead of `self.gamma = nn.Parameter(torch.ones(1))`

## Acknowledgements
- The self attention block implementation is originally by https://github.com/zhaoyuzhi.
- The FID score implementation is by https://github.com/mseitzer/pytorch-fid.
- [Jiahui Yu](https://jiahuiyu.com/) built the original backbone of the GAN implementation.
- This codebase is from the [Fromage repository](https://github.com/jxbz/fromage), developed by Jeremy Bernstein, Arash Vahdat, Yisong Yue & Ming-Yu Liu.
 
## License
This repository (exluding the `fid/` subdirectory) is made available under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

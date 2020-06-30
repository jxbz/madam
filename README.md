<h1 align="center">
Madam optimiser (page under construction)
</h1>

<p align="center">
  <img src="synapse.svg" width="200"/>
</p>

<p align="center">
  <a href="https://jeremybernste.in" target="_blank">Jeremy&nbsp;Bernstein</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://jiawei-zhao.netlify.app" target="_blank">Jiawei&nbsp;Zhao</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.bbe.caltech.edu/people/markus-meister" target="_blank">Markus&nbsp;Meister</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://mingyuliu.net/" target="_blank">Ming&#8209;Yu&nbsp;Liu</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://tensorlab.cms.caltech.edu/users/anima/" target="_blank">Anima&nbsp;Anandkumar</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://www.yisongyue.com" target="_blank">Yisong&nbsp;Yue</a>
</p>

## Getting started

## About this repository

> [Learning compositional functions via multiplicative weight updates](https://arxiv.org/abs/2006.14560).

We're putting this code here so that you can test out our optimisation algorithm in your own applications, and also so that you can attempt to reproduce the experiments in our paper.

If something isn't clear or isn't working, let us know in the *Issues section* or contact [bernstein@caltech.edu](mailto:bernstein@caltech.edu).

## Repository structure

Here is the structure of this repository.

    .
    ├── LICENSE                 # The license on our algorithm.
    └── README.md               # The very page you're reading now.
    
## Acknowledgements

- Our GAN implementation is based on a codebase by [Jiahui Yu](http://jiahuiyu.com/).
- Our Transformer code is from the [Pytorch example](https://github.com/pytorch/examples/tree/master/word_language_model).
- Our CIFAR-10 classification code is orginally by [kuangliu](https://github.com/kuangliu/pytorch-cifar).

## Citation

If you find Madam useful, feel free to cite [the paper](https://arxiv.org/abs/2006.14560):

```bibtex
@misc{madam2020,
    title={Learning compositional functions via multiplicative weight updates},
    author={Jeremy Bernstein and Jiawei Zhao and Markus Meister and Ming-Yu Liu and Anima Anandkumar and Yisong Yue},
    year={2020},
    eprint={arXiv:2002.03432}
}
```

## License

We are making our algorithm available under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. The other code we have used obeys other license restrictions as indicated in the subfolders.

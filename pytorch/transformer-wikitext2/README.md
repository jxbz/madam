<h1 align="center">
Madam optimiser
</h1>

## Wikitext-2 Transformer

This code is forked from the [Pytorch examples](https://github.com/pytorch/examples/tree/master/word_language_model) repository.

Look inside `batch.sh` for the commands used in the paper.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU,
                        Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --optim OPTIM         optimiser
  --levels LEVELS       number of levels for integermadam
  --baselr BASELR       baselr for integermadam
  --decayfactor DECAYFACTOR
                        factor by which to decay learning rate
  --scale SCALE         scale for madam and fromage
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the
                        transformer model
```

# Tune scale in Fromage (this prevents overfitting)
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01 --scale 1.0
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01 --scale 3.0

# Tune lr in SGD
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 0.0001
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 0.001
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 0.01
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 0.1
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 1
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 10

# Tune lr in Adam
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 1.0
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.1
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.01
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.001
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.0001
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.00001

# Tune scale in Madam
python main.py --cuda --epochs 20 --model Transformer --optim madam --lr 0.01 --scale 2.0
python main.py --cuda --epochs 20 --model Transformer --optim madam --lr 0.01 --scale 3.0
python main.py --cuda --epochs 20 --model Transformer --optim madam --lr 0.01 --scale 5.0

# Tune lr in Madam
python main.py --cuda --epochs 10 --model Transformer --optim madam --lr 1.0 --scale 2.0
python main.py --cuda --epochs 10 --model Transformer --optim madam --lr 0.1 --scale 2.0
python main.py --cuda --epochs 10 --model Transformer --optim madam --lr 0.001 --scale 2.0
python main.py --cuda --epochs 10 --model Transformer --optim madam --lr 0.0001 --scale 2.0

# Then tune levels in Integer Madam
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --levels 2048 --scale 2.0
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --levels 4096 --scale 2.0
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --levels 5000 --scale 2.0

# Then tune decay factor in Integer Madam
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --levels 4096 --scale 2.0 --decayfactor 2
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --levels 4096 --scale 2.0 --decayfactor 5

# Run different random seeds for best runs
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01 --scale 1.0 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01 --scale 1.0 --seed 2
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 1 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 1 --seed 2
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.0001 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.0001 --seed 2
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.001 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.001 --seed 2
python main.py --cuda --epochs 20 --model Transformer --optim madam --lr 0.01 --scale 2.0 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim madam --lr 0.01 --scale 2.0 --seed 2

# Integer Madam different bit widths
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --baselr 0.001 --levels 4096 --scale 2.0 --decayfactor 2 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --baselr 0.001 --levels 4096 --scale 2.0 --decayfactor 2 --seed 2

python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --baselr 0.005 --levels 1024 --scale 2.0 --decayfactor 2
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --baselr 0.005 --levels 1024 --scale 2.0 --decayfactor 2 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --baselr 0.005 --levels 1024 --scale 2.0 --decayfactor 2 --seed 2

python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --baselr 0.01 --levels 256 --scale 2.0 --decayfactor 1
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --baselr 0.01 --levels 256 --scale 2.0 --decayfactor 1 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim integermadam --lr 0.01 --baselr 0.01 --levels 256 --scale 2.0 --decayfactor 1 --seed 2

# Fromage tune LR
python main.py --cuda --epochs 10 --model Transformer --optim fromage --lr 1.0 --scale 1.0
python main.py --cuda --epochs 10 --model Transformer --optim fromage --lr 0.1 --scale 1.0
python main.py --cuda --epochs 10 --model Transformer --optim fromage --lr 0.001 --scale 1.0
python main.py --cuda --epochs 10 --model Transformer --optim fromage --lr 0.0001 --scale 1.0
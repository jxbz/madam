# Tune learning rates in each optimiser
python main.py --seed 0 --optim adam --initial_lr 1.0
python main.py --seed 0 --optim adam --initial_lr 0.1
python main.py --seed 0 --optim adam --initial_lr 0.01
python main.py --seed 0 --optim adam --initial_lr 0.001
python main.py --seed 0 --optim adam --initial_lr 0.0001
python main.py --seed 0 --optim adam --initial_lr 0.00001

python main.py --seed 0 --optim sgd --initial_lr 1.0
python main.py --seed 0 --optim sgd --initial_lr 0.1
python main.py --seed 0 --optim sgd --initial_lr 0.01
python main.py --seed 0 --optim sgd --initial_lr 0.001
python main.py --seed 0 --optim sgd --initial_lr 0.0001

python main.py --seed 0 --optim signsgd --initial_lr 0.001
python main.py --seed 0 --optim signsgd --initial_lr 0.0001
python main.py --seed 0 --optim signsgd --initial_lr 0.00001

python main.py --seed 0 --optim madam --initial_lr 1.0
python main.py --seed 0 --optim madam --initial_lr 0.1
python main.py --seed 0 --optim madam --initial_lr 0.01
python main.py --seed 0 --optim madam --initial_lr 0.001
python main.py --seed 0 --optim madam --initial_lr 0.0001

python main.py --seed 0 --optim fromage --initial_lr 1.0
python main.py --seed 0 --optim fromage --initial_lr 0.1
python main.py --seed 0 --optim fromage --initial_lr 0.01
python main.py --seed 0 --optim fromage --initial_lr 0.001
python main.py --seed 0 --optim fromage --initial_lr 0.0001

# Run different random seeds for best setting
python main.py --seed 1 --optim madam --initial_lr 0.01
python main.py --seed 1 --optim fromage --initial_lr 0.01
python main.py --seed 1 --optim adam --initial_lr 0.0001
python main.py --seed 1 --optim sgd --initial_lr 0.01
python main.py --seed 1 --optim signsgd --initial_lr 0.0001

python main.py --seed 2 --optim madam --initial_lr 0.01
python main.py --seed 2 --optim fromage --initial_lr 0.01
python main.py --seed 2 --optim adam --initial_lr 0.0001
python main.py --seed 2 --optim sgd --initial_lr 0.01
python main.py --seed 2 --optim signsgd --initial_lr 0.0001

# B-bit madam experiments. Strategy 1: fix dynamic range and increase base precision.
python main.py --seed 0 --optim intmadam --initial_lr 0.01 --levels 4096 --decayfactor 10
python main.py --seed 0 --optim intmadam --initial_lr 0.01 --levels 2048 --decayfactor 8
python main.py --seed 0 --optim intmadam --initial_lr 0.01 --levels 1024 --decayfactor 4
python main.py --seed 0 --optim intmadam --initial_lr 0.01 --levels 512 --decayfactor 2
python main.py --seed 0 --optim intmadam --initial_lr 0.01 --levels 256 --decayfactor 1

# B-bit madam experiments. Strategy 2: fix base precision and reduce dynamic range.
python main.py --seed 0 --optim intmadam --initial_lr 0.01 --levels 2048 --decayfactor 10
python main.py --seed 0 --optim intmadam --initial_lr 0.01 --levels 1024 --decayfactor 10
python main.py --seed 0 --optim intmadam --initial_lr 0.01 --levels 512 --decayfactor 10
python main.py --seed 0 --optim intmadam --initial_lr 0.01 --levels 256 --decayfactor 10
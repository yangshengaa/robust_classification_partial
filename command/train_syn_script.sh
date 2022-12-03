# generate and train synthetic data

for seed in {1..2000}; do 
    # geerate data 
    python src/data/synthetic.py --seed $seed

    # run continuous training
    # unregularized
    python src/train_syn.py --data syn_cont --model LRRegular --gamma-list 0
    python src/train_syn.py --data syn_cont --model LRRegular --gamma-list 0 --terms 0 
    python src/train_syn.py --data syn_cont --model LRRegular --gamma-list 0 --terms 1 

    python src/train_syn.py --data syn_cont --model SVMRegular --gamma-list 0
    python src/train_syn.py --data syn_cont --model SVMRegular --gamma-list 0 --terms 0 
    python src/train_syn.py --data syn_cont --model SVMRegular --gamma-list 0 --terms 1 

    # regularized / robustified
    python src/train_syn.py --data syn_cont --model LRRegular
    python src/train_syn.py --data syn_cont --model LRRobust
    python src/train_syn.py --data syn_cont --model LRRegular --terms 0
    python src/train_syn.py --data syn_cont --model LRRobust --terms 0
    python src/train_syn.py --data syn_cont --model LRRegular --terms 1
    python src/train_syn.py --data syn_cont --model LRRobust --terms 1

    python src/train_syn.py --data syn_cont --model SVMRegular
    python src/train_syn.py --data syn_cont --model SVMRobust
    python src/train_syn.py --data syn_cont --model SVMRegular --terms 0
    python src/train_syn.py --data syn_cont --model SVMRobust --terms 0
    python src/train_syn.py --data syn_cont --model SVMRegular --terms 1
    python src/train_syn.py --data syn_cont --model SVMRobust --terms 1

    # run discrete training
    # unrgularized
    python src/train_syn.py --data syn_dis --model LRRegular --gamma-list 0 --terms 0
    python src/train_syn.py --data syn_dis --model SVMRegular --gamma-list 0 --terms 0

    # regularized / robustified
    python src/train_syn.py --data syn_dis --model LRRegular --terms 0
    python src/train_syn.py --data syn_dis --model LRRobust --terms 0
    python src/train_syn.py --data syn_dis --model SVMRegular --terms 0
    python src/train_syn.py --data syn_dis --model SVMRobust --terms 0
done 

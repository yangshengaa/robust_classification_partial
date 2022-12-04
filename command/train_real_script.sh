# scripts for training real dataset
for data in "australian" "bands" "heart" "hepatitis" "horse"; do 
    # no regularization
    python src/train_real.py --data $data --model LRRegular --gamma-list 0
    python src/train_real.py --data $data --model SVMRegular --gamma-list 0

    # regularized
    python src/train_real.py --data $data --model LRRegular
    python src/train_real.py --data $data --model SVMRegular

    # robust 
    python src/train_real.py --data $data --model LRRobust
    python src/train_real.py --data $data --model SVMRobust
done 
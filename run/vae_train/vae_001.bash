name=vae_001
flag="--train vae 
      --optim adam --lr 1e-4 --iters 80000"
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name 

name=vae_004
flag="--train vae 
      --optim adam --lr 1e-4 --iters 80000
      --vae_loss_weight 1
      --vae_kl_weight 33
      --vae_bc_weight 1
      "
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name 

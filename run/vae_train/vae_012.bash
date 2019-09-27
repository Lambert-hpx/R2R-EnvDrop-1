name=$(echo $0 | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
flag="--train vae 
      --optim adam --lr 1e-5 --iters 100000
      --vae_loss_weight 1
      --vae_kl_weight 10
      --vae_bc_weight 1
      --vae_latent_dim 16
      "
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name 

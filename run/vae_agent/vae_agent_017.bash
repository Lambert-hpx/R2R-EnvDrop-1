# load agent and vae and finetune
# vae_014
name=$(echo $0 | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
echo $name
flag="--attn soft --train vae_agent
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 
      --maxAction 35
      --load_vae /home/zhufengda/R2R-EnvDrop/snap/vae_014/policy_100000.pkl
      --load snap/vae_agent_010_0/state_dict/best_val_unseen
      "
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name

# modify code of agent_gae_001.bash, retrain
name=$(echo $0 | cut -d. -f1 | cut -d/ -f2)
flag="--attn soft --train vae_agent
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 
      --maxAction 35
      --original_decoder
      --gae
      "
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name 

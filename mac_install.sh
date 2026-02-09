
#!/bin/bash


#conda create -n cs224r python=3.10 -y
#conda activate cs224r

python -m pip install -U pip setuptools wheel

python -m pip install torch torchvision torchaudio
python -m pip install "gymnasium[mujoco]==0.29.1"
python -m pip install "stable-baselines3==2.2.1"
python -m pip install "imitation==1.0.1"
python -m pip install imageio imageio-ffmpeg moviepy tensorboard tqdm ipykernel

python -m ipykernel install --user --name cs224r --display-name "Python (cs224r)"

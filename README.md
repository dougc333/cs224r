Installation directions

Creating expert trajectories requires colab gpus. 
- pick a new gymnasium env like reacher. 
- train using a known baseline, ie PPO from stablebaselines3
- record trajectories. 

This process creates the expert data for BC using  supervised learning to set the  policy. 
Fine tune after BC with PPO.  

How to run cs224r repo under colab: 
Clone the repo under colab to get the datasets
The problem is you have to create a new ssh key everytime a new VM is created. When switching to high-ram CPUs, or adding a GPU will reset the VM. 

1) go to colab, open a terminal, ssh-keygen, copy pub key to github ssh repo key to github/settings enable clones and commits

2) clone to /content. Cloning to /drive/MyDrive/'Colab Notebooks' introduces the FUSE filesystem which is a network file system and not suitable for running colab vms on. 

3) copy back to Colab Notebooks or commit to git repo. 

 

We have 3  envs
1) for macbook conda to run the .mp4 creation from expert data and to use SB3 libs for BC and DAgger. 
macos_make_video_environment.yml

2) requirements.txt and hw1/requirements.txt. Did not use these when creating colab envs for making videos from expert data. 
requirements.txt contains video and python jupyter-book sw. 

2) colab to use the gpu. Colab uses python 3.12 and it doesnt support venv or colab. can install the libs but it causes an error mesage and creating MP4 from expert trajs dont work well after installing mesa libs. 
There is no file for this. These are embedded in the colab cells. Is not clear becuase the install under python 3.12 generates error messages. We have to ignore the error messages. Colab terminal doesnt support venv or colab. 



- replay0.mp4
- replay1.mp4 are 2 trajectories provided in expert_data. Display the video. BC can only get as good as the expert data so BC should still display dead leg syndrome. This is a form of reward hacking; underspecification of constraints. Policy gradient methods can outperform BC. 



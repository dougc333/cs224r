Installation directions

1) go to colab, open a terminal, ssh-keygen, copy pub key to github ssh repo key to enable clones and commits

2) clone to /content. Cloning to /drive/MyDrive/'Colab Notebooks' introduces the FUSE filesystem which is a network file system and not suitable for running colab vms on. 

3) copy back to Colab Notebooks or commit to git repo. 

 
We have 2 envs
1) for macbook conda to run the .mp4 creation from expert data and to use SB3 libs for BC and DAgger. 

2) colab to use the gpu. Colab uses python 3.12 and it doesnt support venv or colab. can install the libs but it causes an error mesage and creating MP4 from expert trajs dont work well after installing mesa libs. 



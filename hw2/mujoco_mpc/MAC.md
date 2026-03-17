 Overview
This codebase makes small modifications to the original MuJoCo MPC software application (https://github.com/deepmind/mujoco_mpc).

### Setup
We assume you are using an AWS EC2 c4.4xlarge instance.
You need to run this line each time you start a new session
`xvfb-run -a -s "-screen 0 1400x900x24" bash`

If you don't, you may seed this error
```
ERROR: could not initialize GLFW

Press Enter to exit ...
```

### Run an example
```
./build/bin/mjpc --task="Quadruped Flat" --steps=100 \
--horizon=0.35 --w0=0.0 --w1=0.0 --w2=0.0 --w3=0.0
```

The expected result should be
```
MuJoCo version 2.3.3
Hardware threads: 16
Agent threads: 13
```
There should also be a new video generated in `mujoco_mpc/videos`.

## Installation
The following should already be installed if you use the AMI image we provide,
but we leave these instructions in case something goes wrong.

###
MacOS install instructions
MacOS has /usr/bin/clang, and /usr/bin/clang++ dont use these. Wrong version
MacOS after homebrew installation has clang-14
/opt/homebrew/bin/clang
/opt/homebrew/bin/clang++
/opt/homebrew/bin/clang-14

These all point to same executable

dc@dcs-MacBook-Pro mujoco_mpc % ls -al /opt/homebrew/Cellar/llvm@14/14.0.6/bin/clang++
lrwxr-xr-x  1 dc  admin  5 Jun 22  2022 /opt/homebrew/Cellar/llvm@14/14.0.6/bin/clang++ -> clang
dc@dcs-MacBook-Pro mujoco_mpc % ls -al /opt/homebrew/Cellar/llvm@14/14.0.6/bin/clang-cpp
lrwxr-xr-x  1 dc  admin  5 Jun 22  2022 /opt/homebrew/Cellar/llvm@14/14.0.6/bin/clang-cpp -> clang


### Clone and build OpenCV
brew install opencv

### Build MJPC
From the `hw2/mujoco_mpc` directory, run:

cmake -B /Users/dc/cs224r/hw2/mujoco_mpc/build -S /Users/dc/cs224r/hw2/mujoco_mpc/mjpc \
  -D CMAKE_C_COMPILER=/opt/homebrew/bin/clang-14 \
  -D CMAKE_CXX_COMPILER=/opt/homebrew/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /Users/dc/cs224r/hw2/mujoco_mpc/build -j4

 Overview

There are no mac install instructions
these are modified after debugging the mujoco_mpc to work on mac



### Run an example
```
./build/bin/mjpc --task="Quadruped Flat" --steps=100 \
--horizon=0.35 --w0=0.0 --w1=0.0 --w2=0.0 --w3=0.0
```

brew install cmake 
brew install zlib
brew install ninja
brew install clang-14? 

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


brew install opencv

### Build MJPC
From the `hw2/mujoco_mpc` directory, run:

cmake -B /Users/dc/cs224r/hw2/mujoco_mpc/build -S /Users/dc/cs224r/hw2/mujoco_mpc/mjpc \
  -D CMAKE_C_COMPILER=/opt/homebrew/bin/clang-14 \
  -D CMAKE_CXX_COMPILER=/opt/homebrew/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /Users/dc/cs224r/hw2/mujoco_mpc/build -j4


You are an automated agent attempting to find a combination of values of w0, w1, w2, w3 to make a quadruped robot walk in a circle. 

Take the following steps iterate and find a combination of weights which cause the robot to walk in a circle;

1) run the mjpc command with the following command line: 

xvfb-run -a -s "-screen 0 1400x900x24" /content/cs224r/hw2/mujoco_mpc/build/bin/mjpc \
  --task="Quadruped Flat" \
  --steps=100 \
  --horizon=0.4 \
  --w0=3.0 -w1=0.0 --w2=0.5 --w3=0.5
to generate an avi video file in the folder videos. The file name will be quadruped_planH_0.400000_w0_3.000000_w1_0.000000_w2_0.500000_w3_0.500000.avi. 
The file name is composed of appending the weight parameters with the values to create the file name. 


2) convert the avi file to a mp4 file using the ffmpeg -i src dst where the src is the filename for the avi file and the dst is the same file name as the avi file name but with a .mp4 extension. As an example the output should be quadruped_planH_0.400000_w0_3.000000_w1_0.000000_w2_0.500000_w3_0.500000.mp4

3) create 4 png files using for t in 0.0 0.1 0.2 0.3; do
for t in 0.0 0.1 0.2 0.3; do
  ffmpeg -ss "$t" -i  -frames:v 1 "frame_${t}.png"
done
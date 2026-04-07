import jax
import mujoco
from mujoco import mjx
import numpy as np

# 1. Define a simple XML model
XML = r"""
<mujoco>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box" pos="0 0 0.2">
            <freejoint/>
            <geom size="0.1 0.1 0.1" type="box" rgba="1 0 0 1"/>
        </body>
        <body name="floor" pos="0 0 0">
            <geom size="1 1 0.1" type="plane" rgba="0.5 0.5 0.5 1"/>
        </body>
    </worldbody>
</mujoco>
"""

# 2. Compile and put model on device
model = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.put_model(model)

# 3. Create data
data = mujoco.MjData(model)
mjx_data = mjx.make_data(mjx_model)

# 4. Define a jitted step function
@jax.jit
def batched_step(mjx_data, mjx_model):
    return mjx.step(mjx_model, mjx_data)

# 5. Run simulation
print("Running simulation...")
for _ in range(100):
    mjx_data = batched_step(mjx_data, mjx_model)
    print(mjx_data.qpos[0]) # Print box position

print("Simulation finished successfully.")


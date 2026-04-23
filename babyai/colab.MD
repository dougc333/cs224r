cd /content/cs224r/babyai
sed -i "s/dtype=np.bool)/dtype=bool)/g" original_babyai_iclr19/babyai/minigrid/minigrid.py
sed -i "s/dtype=np.bool,/dtype=bool,/g" original_babyai_iclr19/babyai/minigrid/minigrid.py



patch all the files np.bool -> bool

find /content/cs224r/babyai/original_babyai_iclr19 -type f -name "*.py" -exec sed -i 's/np.bool/bool/g' {} +

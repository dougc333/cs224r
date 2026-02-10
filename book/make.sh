#!/usr/bin/env bash
set -euo pipefail

#############################################
# Config
#############################################

TARGET_DIR="book/notebooks"
SOURCE_DIR="../../hw1/notebooks"

FILES=(
  ant_policy.ipynb
  ant_rl.ipynb
  cs224r_hw1.ipynb
  read_pkl.ipynb
  loadsafetensors.ipynb
  mnist_rl.ipynb
  mp4_from_expert_data.ipynb
  reacher_expert_traj.ipynb
  sb3_hopper.ipynb
  schema_ant_expert_data.ipynb
  thompson.ipynb
)

#############################################
# Ensure target dir exists
#############################################

mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR"

echo "Creating notebook symlinks..."
echo

#############################################
# Loop through files
#############################################

for f in "${FILES[@]}"; do
    SRC="$SOURCE_DIR/$f"
    DEST="$f"

    # Check source exists
    if [[ ! -f "$SRC" ]]; then
        echo "⚠️  Missing source file: $SRC"
        continue
    fi

    # Remove broken symlink
    if [[ -L "$DEST" && ! -e "$DEST" ]]; then
        echo "🧹 Removing broken symlink: $DEST"
        rm "$DEST"
    fi

    # Skip if real file exists
    if [[ -f "$DEST" && ! -L "$DEST" ]]; then
        echo "⛔ Real file exists, skipping: $DEST"
        continue
    fi

    # Create / update symlink
    if [[ -L "$DEST" ]]; then
        echo "🔁 Updating link: $DEST"
        rm "$DEST"
    else
        echo "🔗 Creating link: $DEST"
    fi

    ln -s "$SRC" "$DEST"
done

echo
echo "✅ Symlink creation complete"

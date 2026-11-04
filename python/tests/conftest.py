import os
import sys
from pathlib import Path

# Resolve directory paths
test_dir = Path(__file__).resolve().parent
python_dir = test_dir.parent
repo_root = python_dir.parent

# 1. Insert python/ into sys.path (for ct_eval package)
python_dir_str = str(python_dir.resolve())
if python_dir_str not in sys.path:
    sys.path.insert(0, python_dir_str)

# 2. Insert the first existing build directory into sys.path (for ct module and ctfilter)
# Order: CT_BUILD_DIR, build/Release, build/Debug, build
candidates = []
env_build = os.environ.get("CT_BUILD_DIR")
if env_build:
    candidates.append(Path(env_build))

candidates.extend([
    repo_root / "build" / "Release",
    repo_root / "build" / "Debug",
    repo_root / "build",
])

for cand in candidates:
    if cand.is_dir():
        cand_str = str(cand.resolve())
        if cand_str not in sys.path:
            sys.path.insert(0, cand_str)
        break

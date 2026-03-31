#!/usr/bin/env bash

set -euo pipefail

python3 - <<'PY'
import sys

if sys.version_info[:2] != (3, 8):
    raise SystemExit(
        f"DART's pip path currently expects Python 3.8.x, got {sys.version.split()[0]}"
    )
PY

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc was not found. The pip installation path builds PyTorch3D from source and expects a local CUDA toolkit." >&2
    exit 1
fi

python3 -m pip install --upgrade pip setuptools wheel

python3 -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

python3 -m pip install \
  Brotli==1.1.0 \
  certifi==2024.2.2 \
  charset-normalizer==3.3.2 \
  colorama==0.4.6 \
  cycler==0.12.1 \
  filelock==3.13.1 \
  fvcore==0.1.5.post20221221 \
  iopath==0.1.9 \
  jinja2==3.1.3 \
  joblib==1.3.2 \
  kiwisolver==1.4.5 \
  markupsafe==2.1.5 \
  matplotlib==3.3.4 \
  networkx==3.1 \
  numpy==1.21.5 \
  packaging==23.2 \
  pandas==2.0.0 \
  patsy==0.5.6 \
  pillow==10.2.0 \
  platformdirs==4.2.0 \
  pooch==1.8.1 \
  portalocker==2.8.2 \
  pyparsing==3.1.2 \
  pysocks==1.7.1 \
  python-dateutil==2.9.0 \
  pytz==2024.1 \
  pyyaml==6.0.1 \
  requests==2.31.0 \
  scikit-learn==1.2.2 \
  scipy==1.10.1 \
  seaborn==0.13.0 \
  six==1.16.0 \
  statsmodels==0.13.5 \
  sympy==1.12 \
  tabulate==0.9.0 \
  termcolor==2.4.0 \
  threadpoolctl==3.3.0 \
  toml==0.10.2 \
  tomli==2.0.1 \
  tqdm==4.66.2 \
  typing_extensions==4.10.0 \
  urllib3==2.2.1 \
  yacs==0.1.8 \
  ninja

python3 -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
python3 -m pip install -r requirements-pip.txt

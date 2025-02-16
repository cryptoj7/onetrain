#!/usr/bin/env bash
export UV_INDEX_STRATEGY=unsafe-any-match
python -m venv venv
source venv/bin/activate
pip install uv
uv pip install --requirement requirements.txt
uv pip install --upgrade --no-deps xformers
uv pip install flash_attn --no-build-isolation
uv pip uninstall onnxruntime
uv pip uninstall xformers
pip install -e git+https://github.com/Nerogar/mgds.git#egg=mgds

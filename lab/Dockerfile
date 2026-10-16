# CUDA 13.2.1 / PyTorch 2.12: the newest NGC release that matches a CUDA 13.2 driver.
# Bump the tag only after `nvidia-smi` on the host reports a CUDA version >= the image's.
ARG BASE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE}
COPY requirements.txt /tmp/lab-requirements.txt
RUN pip install --no-cache-dir -r /tmp/lab-requirements.txt \
 && python -c "import torch; assert 'nv' in torch.__version__, f'pip replaced NGC torch: {torch.__version__}'"
WORKDIR /workspace

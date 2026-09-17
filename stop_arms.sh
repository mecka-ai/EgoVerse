#!/bin/bash
# Stop the arm runs. They all carry batch_size=128, which CUDA-OOMs on H200
# with pi0.5's 3.6B params (135.2 of 139.8 GiB allocated), so every one of them
# dies at its first training step -- no point paying for the rest of staging.
MODAL=/home/mecka/.venv-modal/bin/modal
for app in \
  ap-KYRz9NbWEMjo7FoXccQ3kA \
  ap-paWxUBuQhkKOviMNuG0dFU \
  ap-JwuyZvZwjaAw3T87inrjH1 \
  ap-jallqHjxWeioJh33niPkac \
  ap-w8bpXsjsoPRZbO27qJU34m \
  ap-7wKtRYoKAouCXDG55FTNmu \
  ap-CxqacIBU9xorGWaB2tuZtj \
  ap-59VDAQIFTiEUZ9qebG9ElH ; do
  echo "stopping $app"
  $MODAL app stop "$app" -e robotics --yes 2>&1 | tail -1
done

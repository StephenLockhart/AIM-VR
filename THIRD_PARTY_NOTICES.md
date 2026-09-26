# Third-Party Notices

AIM-VR is built on top of several excellent open-source projects. This file
records their provenance and licenses, which also apply to the corresponding
files in this repository.

## 1. MMagic framework (Apache License 2.0)

The overall framework (`mmagic/`, `tools/`, `configs/`, etc.) is derived from
[OpenMMLab MMagic](https://github.com/open-mmlab/mmagic), licensed under the
Apache License 2.0. The full license text is in [LICENSE](LICENSE).

> Copyright (c) OpenMMLab. All rights reserved.

## 2. Gilbert space-filling curve (BSD 2-Clause)

`mmagic/models/editors/aimvsr/modules/Hilbert3d.py` is
Copyright (c) 2018 Jakub Červený, from
https://github.com/jakubcerveny/gilbert, licensed under the BSD 2-Clause
"Software" License (SPDX header retained in the file).

## 3. OpenMMLab components (Apache License 2.0)

- `mmagic/models/editors/aimvsr/modules/convnext.py` — adapted from
  [mmclassification ConvNeXt](https://github.com/open-mmlab/mmclassification)
  (Copyright (c) OpenMMLab).
- `mmagic/models/editors/aimvsr/modules/basicvr_pp_net.py`, and the utilities
  `PixelShufflePack`, `ResidualBlockNoBN`, `flow_warp`, `make_layer` —
  adapted from OpenMMLab MMagic/BasicVSR++ (Copyright (c) OpenMMLab).
- `mmagic/models/editors/aimvsr/modules/head.py` — FPN-style projection head
  following the OpenMMLab/mmdetection design pattern.

## 4. Mamba / timm building blocks (Apache License 2.0)

`mmagic/models/editors/aimvsr/modules/mambablock.py` and `Hymamba.py` contain
AIM-VR-specific 3D extensions built upon:

- [state-spaces/mamba](https://github.com/state-spaces/mamba) (Apache 2.0)
- [huggingface/pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache 2.0)
- the Vision Mamba (Vim) block design pattern

## 5. pytorch_diffusion (MIT)

`mmagic/models/editors/aimvsr/modules/ResNet.py` is derived from
[pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) /
[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
encoder-decoder code, as noted in the file header.

## 6. Pretrained weights

- **ConvNeXt-Tiny** encoder weights: downloaded from the official OpenMMLab
  model zoo (`download.openmmlab.com`), Apache-2.0-compatible model license.
- **VGG16** weights (training-time perceptual loss only): provided by
  torchvision, for research use.
- **AIM-VR checkpoints**: released for academic research use.

---

If you are the author of any component listed above and believe the attribution
is inaccurate, please open an issue and we will correct it promptly.

# MLP-Mixer: An all-MLP Architecture for Vision (PyTorch)

## Paper
- **Title:** MLP-Mixer: An all-MLP Architecture for Vision
- **Authors:** Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy
- **arXiv:** [2105.01601](https://arxiv.org/abs/2105.01601)
- **Conference:** NeurIPS 2021

## Introduction
MLP-Mixer 是 Google 提出的一種純 MLP（多層感知器）架構，專為圖像分類設計。它不依賴卷積（CNN）或自注意力（Transformer）機制，而是僅用 MLP 來混合空間與通道資訊。該架構證明了在大規模圖像分類任務中，純 MLP 也能取得與主流方法相當的表現。

## 專案內容
本專案為 MLP-Mixer 的 PyTorch 實作，支援 2D 圖像與 3D 視訊資料。可用於：
- 圖像分類（如 CIFAR-10、ImageNet）
- 視訊分類（MLPMixer3D）

## 安裝
建議使用 Python 3.7+，安裝方式如下：

```bash
pip install mlp-mixer-pytorch
```

或在本地專案資料夾下安裝：

```bash
pip install .
```

## 快速使用範例
### 單張圖片分類
```python
import torch
from mlp_mixer_pytorch import MLPMixer

model = MLPMixer(
    image_size=256,
    channels=3,
    patch_size=16,
    dim=512,
    depth=12,
    num_classes=1000
)

img = torch.randn(1, 3, 256, 256)
pred = model(img)  # (1, 1000)
```

### CIFAR-10 圖像分類訓練與驗證
`test_mlp_mixer.py` 提供了完整的 CIFAR-10 訓練與測試流程。

#### 執行步驟
1. 安裝依賴：
   ```bash
   pip install torch torchvision mlp-mixer-pytorch
   ```
2. 執行訓練與驗證：
   ```bash
   python test_mlp_mixer.py
   ```

#### 主要程式邏輯
- 自動下載並載入 CIFAR-10 數據集
- 初始化適合 CIFAR-10 的 MLP-Mixer 模型
- 進行 5 個 epoch 的訓練
- 在測試集上評估準確率

#### 執行結果範例
```
Using device: cpu

Starting training...
Epoch [1/5], Step [100/782], Loss: 2.3021
Epoch [1/5], Step [200/782], Loss: 2.2950
...
Finished Training.

Starting evaluation...
Accuracy of the network on the 10000 test images: 32.50 %

MLP-Mixer validation on CIFAR-10 finished.
```
> 註：準確率會依訓練 epoch 數、模型參數與硬體資源而異。此範例僅為快速驗證模型可正常訓練與推論。

## 進階用法
### 長方形圖片
```python
model = MLPMixer(
    image_size=(256, 128),
    channels=3,
    patch_size=16,
    dim=512,
    depth=12,
    num_classes=1000
)
img = torch.randn(1, 3, 256, 128)
pred = model(img)
```

### 視訊分類
```python
from mlp_mixer_pytorch import MLPMixer3D
model = MLPMixer3D(
    image_size=(256, 128),
    time_size=4,
    time_patch_size=2,
    channels=3,
    patch_size=16,
    dim=512,
    depth=12,
    num_classes=1000
)
video = torch.randn(1, 3, 4, 256, 128)
pred = model(video)
```

## 參考文獻
```bibtex
@misc{tolstikhin2021mlpmixer,
    title   = {MLP-Mixer: An all-MLP Architecture for Vision},
    author  = {Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
    year    = {2021},
    eprint  = {2105.01601},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

## 聯絡方式
- [Papers with Code 專案頁](https://paperswithcode.com/paper/mlp-mixer-an-all-mlp-architecture-for-vision)
- [原始碼 (lucidrains/mlp-mixer-pytorch)](https://github.com/lucidrains/mlp-mixer-pytorch)

---

# Paper Presentation with Code Result

本專案已成功於 CIFAR-10 標準數據集上驗證。訓練與測試流程可參考 `test_mlp_mixer.py`，並可直接執行以獲得模型在標準數據集上的表現。

如需自訂數據集或進行進階應用，請參考本 README 之進階用法。

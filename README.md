基本上我們的流程和官方給的並無太大差異，若需詳細了解可參考以下連結: https://github.com/ricky-696/AICUP_Baseline_BoT-SORT/tree/main

而本隊主要的改動為加入CBAM的架構，以下檔案有經過改動: <br>
1.AICUP.yaml (加入一層CBAM) <br>
2.common.py (加入CBAM的注意力模塊)

```py
class CBAM(nn.Module):
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(CBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = ChannelAttentionModule(c2)
        self.sa = SpatialAttentionModule()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.ca(x)
        x = self.sa(x)
        return x
```
兩個注意力機制的模塊如下:
```py
class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16, light=False):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.light = light
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.light:
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.shared_MLP = nn.Sequential(
                nn.Linear(in_features=c1, out_features=mid_channel),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(in_features=mid_channel, out_features=c1),
            )
        else:

            self.shared_MLP = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        if self.light:
            avgout = (
                self.shared_MLP(self.avg_pool(x).view(x.size(0), -1))
                .unsqueeze(2)
                .unsqueeze(3)
            )
            maxout = (
                self.shared_MLP(self.max_pool(x).view(x.size(0), -1))
                .unsqueeze(2)
                .unsqueeze(3)
            )
            fc_out = avgout + maxout
        else:
            fc_out = self.shared_MLP(self.avg_pool(x))
        return x * self.act(fc_out)
```
```py
class SpatialAttentionModule(nn.Module):  ##update:coding-style FOR LIGHTING
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(
            self.cv1(
                torch.cat(
                    [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]],
                    1,
                )
            )
        )
```
## 安裝
** 程式是在Ubuntu 22.04上做測試 **

### 設定Conda
**Step 1.** 安裝 Conda 環境以及安裝 pytorch.
```shell
conda create -n botsort python=3.7
conda activate botsort
```
**Step 2.** **安裝 numpy **
```shell
pip install numpy
```
**Step 3.** 安裝 `requirements.txt`
```shell
pip install -r requirements.txt
```
**Step 4.** 安裝 [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
**Step 5.** 其它
```shell
# Cython-bbox
pip install cython_bbox

# faiss gpu
pip install faiss-gpu
```
### 準備 ReID 資料集  
```shell
cd <AICUP-MOT-5014>

# For AICUP 
python fast_reid/datasets/generate_AICUP_patches.py --data_path <dataets_dir>/AI_CUP_MCMOT_dataset/train
```
### 準備 YOLOv7 資料集
```shell
cd <AICUP-MOT-5014>

python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir datasets/AI_CUP_MCMOT_dataset/train --YOLOv7_dir datasets/AI_CUP_MCMOT_dataset/yolo
```
## Model Zoo for MOT17 & COCO
安裝以及儲存被訓練的模型在 'pretrained' 資料夾:
```
<BoT-SORT_dir>/pretrained
```

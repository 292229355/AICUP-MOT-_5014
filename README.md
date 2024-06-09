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

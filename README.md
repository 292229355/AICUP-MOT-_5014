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

import torch
import torch.nn as nn






MHA = torch.nn.MultiheadAttention(32,4,batch_first=True)

B  = 8
lidarpoints = torch.rand((B,4,16,32))
radarpoints = torch.rand((B,4,16,32))

ret = []
for b in range(B):
    out = MHA(radarpoints[b],lidarpoints[b],lidarpoints[b])
    ret += [out[0]]
# lidarpoints2 = torch.rand((B,24,32))
# radarpoints2 = torch.rand((B,24,32))


# out2 = MHA(radarpoints2,lidarpoints2,lidarpoints2)



print(ret)
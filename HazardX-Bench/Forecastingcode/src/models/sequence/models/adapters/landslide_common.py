from __future__ import annotations
import torch

def physical(batch,order=("phi","z_norm","h0_norm")):
    return torch.stack(tuple(batch[k] for k in order),dim=1)
def explicit_sequence(batch):
    b=batch["phi"].shape[0]; static=physical(batch).unsqueeze(2).expand(-1,-1,9,-1,-1)
    x=batch["x_grid"].unsqueeze(1).unsqueeze(2).expand(-1,1,9,-1,-1)
    y=batch["y_grid"].unsqueeze(1).unsqueeze(2).expand_as(x)
    t=batch["t_grid"].reshape(b,1,9,1,1).expand(-1,1,-1,200,200)
    return torch.cat((x,y,t,static),1)
def assert_output(y,batch):
    expected=(batch["phi"].shape[0],8,2,200,200)
    if tuple(y.shape)!=expected: raise AssertionError(f"{tuple(y.shape)} != {expected}")
    if not torch.isfinite(y).all(): raise FloatingPointError("non-finite prediction")
    return y
class LpLoss:
    def __init__(self,size_average=True,reduction=True,eps=1e-12): self.size_average=size_average; self.reduction=reduction; self.eps=eps
    def __call__(self,x,y):
        n=x.shape[0]; d=torch.norm((x-y).reshape(n,-1),2,1); q=torch.norm(y.reshape(n,-1),2,1); v=d/(q+self.eps)
        return v.mean() if self.size_average else v.sum()
def benchmark_loss(pred,target):
    assert pred.shape==target.shape and pred.shape[1:3]==(8,2)
    loss=LpLoss(); h=loss(pred[:,:,0],target[:,:,0]); u=loss(pred[:,:,1],target[:,:,1]); return 7*h+u,h,u


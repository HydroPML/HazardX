import torch
import torch.nn as nn
from gphyt.model.convlstm_openstl import ConvLSTMCell
from .landslide_common import explicit_sequence,assert_output
class LandslideConvLSTM(nn.Module):
    def __init__(self,channels_last=True,**_):
        super().__init__()
        self.channels_last=bool(channels_last)
        widths=(128,)*4
        self.cells=nn.ModuleList(ConvLSTMCell(6 if i==0 else 128,c,5) for i,c in enumerate(widths))
        self.recovery=nn.Conv2d(128,2,1)
        if self.channels_last:
            self.to(memory_format=torch.channels_last)
    def _state(self,seq,batch_size,height,width):
        value=seq.new_zeros(batch_size,128,height,width)
        if self.channels_last:
            value=value.contiguous(memory_format=torch.channels_last)
        return value
    def forward(self,batch):
        seq=explicit_sequence(batch).permute(0,2,1,3,4)
        b,_,_,h,w=seq.shape
        states=[(self._state(seq,b,h,w),self._state(seq,b,h,w)) for _ in self.cells]
        out=[]
        for ti in range(9):
            hidden=seq[:,ti]
            if self.channels_last:
                hidden=hidden.contiguous(memory_format=torch.channels_last)
            nxt=[]
            for cell,state in zip(self.cells,states):
                hidden,cell_state=cell(hidden,state); nxt.append((hidden,cell_state))
            states=nxt; out.append(self.recovery(hidden))
        return assert_output(torch.stack(out[1:],1),batch)

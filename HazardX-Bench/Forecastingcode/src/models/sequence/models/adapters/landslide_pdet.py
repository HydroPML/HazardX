"""Separate-channel PDE-Transformer SC-S with native query-time metadata."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gphyt.model.pde_transformer_sc_s import PDETransformerSCSPTFT
from .landslide_common import physical,assert_output

class LandslidePDET(nn.Module):
    spatial_padding=(56,56)
    def __init__(self,official_root="",checkpoint_path="",report_path="",lead_chunk=8,**_):
        super().__init__()
        self.lead_chunk=max(1,min(8,int(lead_chunk)))
        self.core=PDETransformerSCSPTFT(256,256,official_root=official_root,checkpoint_path=checkpoint_path,report_path=report_path)
        self.recovery=nn.Conv2d(3,2,1)
        self.loading_report=self.core.loading_report
        for name in ("recovery.weight","recovery.bias"):
            if name not in self.loading_report.newly_initialized: self.loading_report.newly_initialized.append(name)
        self.loading_report.newly_initialized.sort(); self.loading_report.write(report_path)
    def forward(self,batch):
        physical_fields=F.pad(physical(batch,("z_norm","phi","h0_norm")),(0,56,0,56))
        batch_size=physical_fields.shape[0]; predictions=[]
        leads=torch.linspace(.125,1.,8,device=physical_fields.device,dtype=physical_fields.dtype)
        for start in range(0,8,self.lead_chunk):
            chunk=leads[start:start+self.lead_chunk]; chunk_size=chunk.numel()
            expanded=physical_fields[:,None].expand(batch_size,chunk_size,-1,-1,-1).reshape(batch_size*chunk_size,3,256,256)
            fields=[torch.stack((expanded[:,i],torch.zeros_like(expanded[:,i])),1) for i in range(3)]
            zeros,channel_type,labels,parameters,parameter_classes,task=self.core._per_channel_metadata(batch_size*chunk_size,self.core.channel_type_ids,physical_fields.device,physical_fields.dtype)
            lead_batch=chunk[None].expand(batch_size,chunk_size).reshape(-1)
            query=[lead_batch for _ in zeros]
            result=self.core.backbone(x=fields,simulation_time=query,channel_type=channel_type,pde_type=labels,pde_parameters=parameters,pde_parameters_class=parameter_classes,simulation_dt=query,task=task,t=query,return_dict=True)
            outputs=result.sample
            if not isinstance(outputs,list) or len(outputs)!=3: raise AssertionError("SC-S returned invalid channel list")
            state=torch.cat([value[:,1:2] for value in outputs],1)
            prediction=self.recovery(state)[:,:,:200,:200].reshape(batch_size,chunk_size,2,200,200)
            predictions.append(prediction)
        return assert_output(torch.cat(predictions,1),batch)

import torch
from CUSTOMLIB/torch import MODEL
from thop import profile

device = 'cuda:0'
model = MODEL().to(device)
input_data = torch.rand(2,dim1,dim2,dim3).to(device)
flops, params = profile(model,inputs=(input_data,),verbose=False)
print("Number of FLOPs (in B)", flops/1000000000)
print("Number of Learnable Params (in M)", params/1000000)

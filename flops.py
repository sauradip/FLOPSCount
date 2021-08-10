import torch
from gsm_model import GSM


# Count the number of FLOPs
# count_ops(model, inp)

from thop import profile

# model = resnet50 ()
device = 'cuda:0'
model = GSM().to(device)
input_data = torch.rand(2,2048,100).to(device)
flops, params = profile(model,inputs=(input_data,),verbose=False)
print("flops",flops/1000000000)
# # import torchvision.models as models
# import torch
# from ptflops import get_model_complexity_info

# with torch.cuda.device(0):
# #   net = models.densenet161()
#   net = GSM()
#   macs, params = get_model_complexity_info(net, (2, 2048, 100), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))
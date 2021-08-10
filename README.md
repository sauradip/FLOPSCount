# FLOPSCount

A cute script to count FLOPS and number of learnable PARAMS for any custom or pre-defined network

* Step 1 : Cloning and Environment Setup : 

```bash
    git clone https://github.com/sauradip/FLOPSCount.git
    cd FLOPSCount/
    pip3 install thop
   ```
   
* Step 2 : Calculating Metrics for Custom or Predefined Model : 

> Set the "library-path"  of your "Model()" in Line 2, Model() should contain torch.xx
> Set the "device-id" of your system in Line 5, ideally GPU 
> Set the "dimension" of the input tensor of your "Model()"



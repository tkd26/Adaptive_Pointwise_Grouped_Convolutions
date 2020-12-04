#this class is trying to do the same thig as the author's implementation
# https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L242-L294 

import torch
import torchvision
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class AdaBIGGAN(nn.Module):
    def __init__(self,generator, dataset_size, embed_dim=120, shared_embed_dim = 128,cond_embed_dim = 20,embedding_init="zero"):
        '''
        generator: original big gan generator
        dataset_size: (small) number of training images. It should be less than 100. If more than 100, it's better to fine tune using normal adverserial training
        shared_embed_dim: class shared embedding dim. 
        cond_embed_dim: class conditional embedding dim
        See Generator row 2 in table 4 in the BigGAN paper (1809.11096v2) where Linear(20+129), which means Linear(cond_embed_dim+shared_embed_dim) 
        '''
        super(AdaBIGGAN,self).__init__()

        self.generator = generator
        #same as z in the chainer implementation
        self.embeddings = nn.Embedding(dataset_size, embed_dim)
        if embedding_init == "zero":
            self.embeddings.from_pretrained(torch.zeros(dataset_size,embed_dim),freeze=False)
        
        # in_channels = self.generator.blocks[0][0].conv1.in_channels
        # self.bsa_linear_scale = torch.nn.Parameter(torch.ones(in_channels,))
        # self.bsa_linear_bias = torch.nn.Parameter(torch.zeros(in_channels,))
        
        self.linear = nn.Linear(1, shared_embed_dim, bias=False)
        #torch.nn.init.kaiming_normal_(self.linear.weight)
        init_weight = generator.shared.weight.mean(dim=0,keepdim=True).transpose(1,0)
        assert self.linear.weight.data.shape == init_weight.shape
        self.linear.weight.data  = init_weight
        del generator.shared


        # blockのconv1x1
        self.conv1x1 = []
        for ch in [1536, 1536, 1536, 768, 768, 384, 384, 192, 192, 96]:
            conv = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0).cuda()
            conv = weight_norm(conv)
            # torch.nn.init.xavier_uniform_(conv.weight)
            torch.nn.init.kaiming_normal_(conv.weight)
            # conv.weight.data.fill_(1.)
            # conv.bias.data.fill_(0)
            self.conv1x1 += [conv]
        self.conv1x1 = nn.ModuleList(self.conv1x1)

        i = 0
        for index, blocklist in enumerate(self.generator.blocks):
            for block in blocklist:
                try:
                    block.activation1 =  nn.Sequential(
                        self.conv1x1[i], 
                        block.activation1
                    )
                    block.activation2 =  nn.Sequential(
                        self.conv1x1[i+1],
                        block.activation2
                    )
                    i += 2
                except:
                    continue

        # 最初のレイヤのconv1x1
        in_ch = self.generator.blocks[0][0].conv1.in_channels
        self.conv1x1_first = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0).cuda()
        self.conv1x1_first = weight_norm(self.conv1x1_first)
        # torch.nn.init.xavier_uniform_(self.conv1x1_first.weight)
        torch.nn.init.kaiming_normal_(self.conv1x1_first.weight)
        # self.conv1x1_first.weight.data.fill_(1.)
        # self.conv1x1_first.bias.data.fill_(0)
        
        self.set_training_parameters()
                
    def forward(self, z):
        '''
        z: tensor whose shape is (batch_size, shared_embed_dim) . in the training time noise (`epsilon` in the original paper) should be added. 
        '''
        #originally copied from the biggan repo
        #https://github.com/ajbrock/BigGAN-PyTorch/blob/ba3d05754120e9d3b68313ec7b0f9833fc5ee8bc/BigGAN.py#L226-L251
        #then modified to do the same job in chainer smallgan repo
        #https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L278-L294

        #original note, as original one use `forward(self, z, y)` (notice y)
        ##Note on this forward function: we pass in a y vector which has
        ##already been passed through G.shared to enable easy class-wise
        ##interpolation later. If we passed in the one-hot and then ran it through
        ##G.shared in this forward function, it would be harder to handle.

        #my note
        #here, we *do* make `y` inside forwad function
        #`y` is equivalent to `c` in chainer smallgan repo

        y = torch.ones((z.shape[0], 1),dtype=torch.float32,device=z.device)#z.shape[0] is batch size
        y = self.linear(y)

        # If hierarchical (i.e. use different z per layer), concatenate zs and ys
        if self.generator.hier:
            zs = torch.split(z, self.generator.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            raise NotImplementedError("I don't implement this case")
            ys = [y] * len(self.generator.blocks)

        # First linear layer
        h = self.generator.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.generator.bottom_width, self.generator.bottom_width)
        h = self.conv1x1_first(h)
        
        #Do scale and bias (i.e. apply newly intoroduced statistic parameters) for the first linear layer
        # h = h*self.bsa_linear_scale.view(1,-1,1,1) + self.bsa_linear_bias.view(1,-1,1,1) 
        
        # Loop over blocks
        for index, blocklist in enumerate(self.generator.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.generator.output_layer(h))
    

    
    def set_training_parameters(self):
        '''
        set requires_grad=True only for parameters to be updated, requires_grad=False for others.
        '''
        #set all parameters requires_grad=False first
        for param in self.parameters():
            param.requires_grad = False
            
        named_params_requires_grad = {}
        # named_params_requires_grad.update(self.batch_stat_gen_params())
        named_params_requires_grad.update(self.linear_gen_params())
        # named_params_requires_grad.update(self.bsa_linear_params())
        named_params_requires_grad.update(self.calss_conditional_embeddings_params())
        named_params_requires_grad.update(self.emebeddings_params())
        named_params_requires_grad.update(self.conv1x1_params())
        named_params_requires_grad.update(self.conv1x1_first_params())
        
        for name,param in named_params_requires_grad.items():
            param.requires_grad = True

    def conv1x1_first_params(self):
        # return {"conv1x1_first.weight":self.conv1x1_first.weight,"conv1x1_first.bias":self.conv1x1_first.bias}
        return {
            "conv1x1_first.weight_g":self.conv1x1_first.weight_g,
            "conv1x1_first.weight_v":self.conv1x1_first.weight_v,
            "conv1x1_first.bias":self.conv1x1_first.bias}


    def conv1x1_params(self):
        '''
        conv1x1.0.weight
        conv1x1.0.bias
        conv1x1.1.weight
        conv1x1.1.bias
        conv1x1.2.weight
        conv1x1.2.bias
        conv1x1.3.weight
        conv1x1.3.bias
        conv1x1.4.weight
        conv1x1.4.bias
        conv1x1.5.weight
        conv1x1.5.bias
        conv1x1.6.weight
        conv1x1.6.bias
        conv1x1.7.weight
        conv1x1.7.bias
        conv1x1.8.weight
        conv1x1.8.bias
        conv1x1.9.weight
        conv1x1.9.bias
        '''
        named_params = {}
        for name, value in self.conv1x1.named_parameters():
            name = 'conv1x1.' + name
            # print(name)
            named_params[name] = value
        return named_params

            
    # def batch_stat_gen_params(self):
    #     '''
    #     get named parameters to generate batch statistics
    #     Weight corresponding to "Hyper" in Chainer implementation 
    #     ```
    #         blocks.0.0.bn1.gain.weight torch.Size([1536, 148])
    #         blocks.0.0.bn1.bias.weight torch.Size([1536, 148])
    #         blocks.0.0.bn2.gain.weight torch.Size([1536, 148])
    #         blocks.0.0.bn2.bias.weight torch.Size([1536, 148])
    #         blocks.1.0.bn1.gain.weight torch.Size([1536, 148])
    #         blocks.1.0.bn1.bias.weight torch.Size([1536, 148])
    #         blocks.1.0.bn2.gain.weight torch.Size([768, 148])
    #         blocks.1.0.bn2.bias.weight torch.Size([768, 148])
    #         blocks.2.0.bn1.gain.weight torch.Size([768, 148])
    #         blocks.2.0.bn1.bias.weight torch.Size([768, 148])
    #         blocks.2.0.bn2.gain.weight torch.Size([384, 148])
    #         blocks.2.0.bn2.bias.weight torch.Size([384, 148])
    #         blocks.3.0.bn1.gain.weight torch.Size([384, 148])
    #         blocks.3.0.bn1.bias.weight torch.Size([384, 148])
    #         blocks.3.0.bn2.gain.weight torch.Size([192, 148])
    #         blocks.3.0.bn2.bias.weight torch.Size([192, 148])
    #         blocks.4.0.bn1.gain.weight torch.Size([192, 148])
    #         blocks.4.0.bn1.bias.weight torch.Size([192, 148])
    #         blocks.4.0.bn2.gain.weight torch.Size([96, 148])
    #         blocks.4.0.bn2.bias.weight torch.Size([96, 148])
    #     ```
    #     '''
    #     named_params = {}
    #     for name,value in self.named_modules():
    #         if name.split(".")[-1] in ["gain","bias"]:
    #             for name2,value2 in  value.named_parameters():
    #                 name = name+"."+name2
    #                 params = value2
    #                 named_params[name] = params
                    
    #     return named_params
       
    def linear_gen_params(self):
        '''
        Fully connected weights in generator
        finetune with very small learning rate
        ```
            linear.weight torch.Size([24576, 20])
            linear.bias torch.Size([24576])
        ```
        '''
        return {"generator.linear.weight":self.generator.linear.weight,
                       "generator.linear.bias":self.generator.linear.bias}

    # def bsa_linear_params(self):
    #     '''
    #     Statistics parameter (scale and bias) after lienar layer
    #     This is a newly intoroduced training parameters that did not exist in the original generator
    #     '''
    #     return {"bsa_linear_scale":self.bsa_linear_scale,"bsa_linear_bias":self.bsa_linear_bias}

    def calss_conditional_embeddings_params(self):
        '''
        128 dim input as the conditional noise (?)
        '''
        return {"linear.weight":self.linear.weight}


    def emebeddings_params(self):
        '''
        initialized with zero but added with random epsilon for training time
        this is 120 in the BigGAN 128 x 128 while 140 in 256 x 256
        '''
        return  {"embeddings.weight":self.embeddings.weight}

    
if __name__ == "__main__":
    import sys
    sys.path.append("../official_biggan_pytorch/")
    sys.path.append("../")
    from official_biggan_pytorch import utils
    
    import torch
    import torchvision

    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args(args=[]))
    
    # taken from https://github.com/ajbrock/BigGAN-PyTorch/issues/8
    config["resolution"] = utils.imsize_dict["I128_hdf5"]
    config["n_classes"] = utils.nclass_dict["I128_hdf5"]
    config["G_activation"] = utils.activation_dict["inplace_relu"]
    config["D_activation"] = utils.activation_dict["inplace_relu"]
    config["G_attn"] = "64"
    config["D_attn"] = "64"
    config["G_ch"] = 96
    config["D_ch"] = 96
    config["hier"] = True
    config["dim_z"] = 120
    config["shared_dim"] = 128
    config["G_shared"] = True
    config = utils.update_config_roots(config)
    config["skip_init"] = True
    config["no_optim"] = True
    config["device"] = "cuda"

    # Seed RNG.
    utils.seed_rng(config["seed"])

    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True

    # Import the model.
    model = __import__(config["model"])
    experiment_name = utils.name_from_config(config)
    G = model.Generator(**config).to(config["device"])
    utils.count_parameters(G)

    # Load weights.
    weights_path = "../data/G_ema.pth"  # Change this.
    # weights_path = "./data/G.pth"  # Change this.
    G.load_state_dict(torch.load(weights_path))

    model = AdaBIGGAN(G,dataset_size=42)
    model = model.cuda()
    # print(model)
    
    batch_size = 4
    
    z = torch.ones((batch_size,140)).cuda()

    output = model(z)
    
    assert output.shape == (batch_size,3,128,128)
    
    print("simple test pased!")
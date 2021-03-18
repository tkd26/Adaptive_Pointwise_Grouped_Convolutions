import os
import argparse
import time
import numpy as np
import random

from scipy.stats import truncnorm

import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler

# imports from my own script
from utils import gpu_setup,savedir_setup,save_args,save_json,check_githash,save_checkpoint
import visualizers
from metrics.AverageMeter import AverageMeter
from dataloaders.setup_dataloader_smallgan import setup_dataloader
from models.setup_model import setup_model
from loss.AdaBIGGANLoss import AdaBIGGANLoss
from loss.KMMD import KMMD

SEED = 1
def define_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    global SEED
    SEED += 1

def argparse_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="anime", help = "dataset. anime or face or flower. ")
    parser.add_argument('--pretrained', type=str, default="./data/G_ema.pth", help = "pretrained BigGAN model")
    parser.add_argument('--mode', type=str, default="train", help = "mode")
    parser.add_argument('--KMMD', action='store_true')
    parser.add_argument('--FID', action='store_true')

    parser.add_argument('--eval-freq', type=int, default=500, help = "save frequency in iteration. currently no eval is implemented and just model saving and sample generation is performed" )
    parser.add_argument('--gpu', '-g', type=str, default='-1')
    
    #learning rates
    parser.add_argument('--lr-g-l', type=float, default=0.0000001, help = "lr for original linear layer in generator. default is 0.0000001")
    parser.add_argument('--lr-g-batch-stat', type=float, default=0.0005, help = "lr for linear layer to generate scale and bias in generator")
    parser.add_argument('--lr-embed', type=float, default=0.05, help = "lr for image embeddings")
    parser.add_argument('--lr-bsa-l', type=float, default=0.0005, help = "lr for statistic (scale and bias) parameter for the original fc layer in generator. This is newly intoroduced learnable parameter for the pretrained GAN")
    parser.add_argument('--lr-c-embed', type=float, default=0.001, help = "lr for class conditional embeddings")
    
    #loss settings
    parser.add_argument('--loss-per', type=float, default=0.1, help = "scaling factor for perceptural loss. ")
    parser.add_argument('--loss-emd', type=float, default=0.1, help = "scaling factor for earth mover distance loss. ")
    parser.add_argument('--loss-re', type=float, default=0.02, help = "scaling factor for regularizer loss. ")
    parser.add_argument('--loss-norm-img', type=int, default=1, help = "normalize img loss or not")
    parser.add_argument('--loss-norm-per', type=int, default=1, help = "normalize perceptural loss or not")
    parser.add_argument('--loss-dist-per', type=str, default="l2", help = "distance function for perceptual loss. l1 or l2")

    parser.add_argument('--step', type=int, default=3000, help="Decrease lr by a factor of args.step_facter every <step> iterations")
    parser.add_argument('--step-facter', type=float, default=0.1, help="facter to multipy when decrease lr ")

    parser.add_argument('--epoch', type=int, default=30000, help="number of epochs.")
    parser.add_argument('--batch', type=int, default=25, help="batch size")
    parser.add_argument('--workers', type=int, default=4, help="number of processes to make batch worker. default is 8")
    parser.add_argument('--data_num', type=int, default=50, help="")
    parser.add_argument('--model', type=str,default = "biggan128-ada", help = "model. biggan128-ada or biggan128-conv1x1-3")
    parser.add_argument('--groups', type=int, default=1, help="")
    parser.add_argument('--per_groups', type=int, default=0, help="ch/per_groups")

    parser.add_argument('--resume', type=str, default=None, help="model weights to resume")
    parser.add_argument('--savedir',  default = "train", help='Output directory')
    parser.add_argument('--saveroot',  default = "./experiments", help='Root directory to make the output directory')

    parser.add_argument('-p', '--print-freq', default=100, type=int, help='print frequency ')
    return parser.parse_args()

def generate_samples(model,img_prefix,batch_size):
    visualizers.reconstruct(model,img_prefix+"reconstruct.jpg",torch.arange(batch_size),True)
    visualizers.interpolate(model,img_prefix+"interpolate.jpg",source=0,dist=1,trncate=0.3, num=7)
    visualizers.random(model,img_prefix+"random.jpg",tmp=0.3, n=9, truncate=True)

def setup_optimizer(model_name,model,lr_g_batch_stat,lr_g_linear,lr_bsa_linear,lr_embed,lr_class_cond_embed,step,step_facter=0.1):
    #group parameters by lr
    params = []
    if model_name=='biggan128-ada': # FiLM
        params.append({"params":list(model.batch_stat_gen_params().values()), "lr":lr_g_batch_stat})
        params.append({"params":list(model.linear_gen_params().values()), "lr":lr_g_linear })
        params.append({"params":list(model.bsa_linear_params().values()), "lr":lr_bsa_linear })
        params.append({"params":list(model.emebeddings_params().values()), "lr": lr_embed })
        params.append({"params":list(model.calss_conditional_embeddings_params().values()), "lr":lr_class_cond_embed})

    # elif model_name=='biggan128-conv1x1': # 旧モデル
    #     params.append({"params":list(model.conv1x1_params().values()), "lr": lr_g_batch_stat })
    #     params.append({"params":list(model.conv1x1_first_params().values()), "lr": lr_bsa_linear })
    #     params.append({"params":list(model.linear_gen_params().values()), "lr":lr_g_linear }) # lr_g_linear
    #     params.append({"params":list(model.embeddings_params().values()), "lr": 0.01 })
    #     # params.append({"params":list(model.calss_conditional_embeddings_params().values()), "lr":lr_class_cond_embed})

    # elif model_name=='biggan128-conv1x1-2': # 旧モデル
    #     # params.append({"params":list(model.conv1x1_params().values()), "lr": lr_g_batch_stat })
    #     # params.append({"params":list(model.conv1x1_first_params().values()), "lr": lr_bsa_linear })
    #     params.append({"params":list(model.linear_gen_params().values()), "lr":lr_g_linear }) # lr_g_linear
    #     params.append({"params":list(model.embeddings_params().values()), "lr": 0.1 })
    #     # params.append({"params":list(model.calss_conditional_embeddings_params().values()), "lr":lr_class_cond_embed})
    #     params.append({"params":list(model.conv1x1_paramG_weights_params().values()), "lr": lr_bsa_linear })
    #     params.append({"params":list(model.conv1x1_paramG_biases_params().values()), "lr": lr_bsa_linear })

    elif model_name=='biggan128-conv1x1-3': # メインで使用しているモデル
        params.append({"params":list(model.linear_gen_params().values()), "lr":lr_g_linear }) # lr_g_linear
        params.append({"params":list(model.embeddings_params().values()), "lr": lr_embed })
        params.append({"params":list(model.calss_conditional_embeddings_params().values()), "lr":lr_class_cond_embed})
        params.append({"params":list(model.conv1x1_paramG_weights_params().values()), "lr": lr_g_batch_stat })
        params.append({"params":list(model.conv1x1_paramG_biases_params().values()), "lr": lr_g_batch_stat })
        # params.append({"params":list(model.conv1x1_first_paramG_weight_params().values()), "lr": lr_bsa_linear })
        # params.append({"params":list(model.conv1x1_first_paramG_bias_params().values()), "lr": lr_bsa_linear })
        params.append({"params":list(model.bsa_linear_params().values()), "lr":lr_bsa_linear })

    #setup optimizer
    optimizer = optim.Adam(params, lr=0)#0 is okay because sepcific lr is set by `params`
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=step_facter)
    return optimizer,scheduler

def main(args):
    device = gpu_setup(args.gpu)
    append_args = ["dataset","data_num","model"]
    if args.mode == 'train':
        checkpoint_dir = savedir_setup(args.savedir,args=args,append_args=append_args,basedir=args.saveroot)
        args.githash = check_githash()
        save_args(checkpoint_dir,args)
    
    dataloader = setup_dataloader(name=args.dataset,
                                   batch_size=args.batch,
                                   num_workers=args.workers,
                                   data_num=args.data_num,
                                  )
    
    dataset_size = len(dataloader.dataset)
    print("number of images (dataset size): ",dataset_size)
    
    if args.model == "biggan128-ada":
        model = setup_model(args.model,dataset_size=dataset_size,resume=args.resume,biggan_imagenet_pretrained_model_path=args.pretrained)
    elif args.model == "biggan128-conv1x1":
        model = setup_model(args.model,dataset_size=dataset_size,resume=args.resume,biggan_imagenet_pretrained_model_path=args.pretrained)
    elif args.model == 'biggan128-conv1x1-2':
        model = setup_model(args.model,dataset_size=dataset_size,resume=args.resume,biggan_imagenet_pretrained_model_path=args.pretrained,groups=args.groups)
    elif args.model == 'biggan128-conv1x1-3':
        model = setup_model(args.model,dataset_size=dataset_size,resume=args.resume,biggan_imagenet_pretrained_model_path=args.pretrained,per_groups=args.per_groups)
    else:
        print('Error: Model not defined')
        sys.exit(1)
    model.eval()
    #this has to be eval() even if it's training time
    #because we want to fix batchnorm running mean and var
    #still tune batchnrom scale and bias that is generated by linear layer in biggan
    
    optimizer,scheduler = setup_optimizer(model_name = args.model,
                                          model = model,
                                          lr_g_linear = args.lr_g_l,
                                          lr_g_batch_stat  = args.lr_g_batch_stat,
                                          lr_bsa_linear  = args.lr_bsa_l,
                                          lr_embed = args.lr_embed,
                                          lr_class_cond_embed = args.lr_c_embed,
                                          step= args.step,
                                          step_facter = args.step_facter,
                                         )
    
    criterion = AdaBIGGANLoss(
                    scale_per=args.loss_per,
                    scale_emd=args.loss_emd,
                    scale_reg=args.loss_re,
                    normalize_img = args.loss_norm_img,
                    normalize_per = args.loss_norm_per,
                    dist_per = args.loss_dist_per,
                )
    
    #start trainig loop
    losses = AverageMeter()
    eval_kmmd = AverageMeter()
    print_freq = args.print_freq
    eval_freq = args.eval_freq
    save_freq = eval_freq
    max_epoch = args.epoch
    log = {}
    log["log"]=[]
    since = time.time()

    min_loss = {
        'data': 1000,
        'epoch': 0,
    }
    
    epoch = 0
    #prepare model and loss into device
    model = model.to(device)
    criterion = criterion.to(device)
    while(True):

        if args.mode == 'train': scheduler.step()

        # Iterate over dataset (one epoch).
        for data in dataloader: 
            define_seed(SEED)
            img = data[0].to(device)
            indices = data[1].to(device)
        
            #embeddings (i.e. z) + noise (i.e. epsilon) 
            embeddings = model.embeddings(indices)
            embeddings_eps = torch.randn(embeddings.size(),device=device)*0.01
            #see https://github.com/nogu-atsu/SmallGAN/blob/f604cd17516963d8eec292f3faddd70c227b609a/gen_models/ada_generator.py#L29
            #forward
            img_generated = model(embeddings+embeddings_eps)

            if args.mode == 'train':
                loss = criterion(img_generated,img,embeddings,model.linear.weight)
                losses.update(loss.item(), img.size(0))
                #compute gradient and do SGD step
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

            elif args.mode == 'eval':
                if args.KMMD:
                    # KMMD
                    latent_size = embeddings.size(1)
                    true_sample = torch.randn(args.batch, latent_size, requires_grad = False).to(device)
                    kmmd = KMMD()(true_sample, embeddings)
                    eval_kmmd.update(kmmd.item(), img.size(0))

        if args.mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % print_freq == 0:
                if min_loss['data'] > losses.avg:
                    min_loss['data'] = losses.avg
                    min_loss['epoch'] = epoch
                temp = "train loss: %0.5f "%loss.item()
                temp += "| smoothed loss %0.5f "%losses.avg
                temp += "| min loss %0.5f (%d)"%(min_loss['data'], min_loss['epoch'])
                log["log"].append({"epoch":epoch,"loss":losses.avg})
                print(epoch,temp)
                losses = AverageMeter()
            if epoch % eval_freq==0:
                img_prefix = os.path.join(checkpoint_dir,"%d_"%epoch) 
                generate_samples(model,img_prefix,dataloader.batch_size)
                
            if epoch % save_freq==0:
                save_checkpoint(checkpoint_dir,device,model,iteration=epoch )

        elif args.mode == 'eval':
            if epoch * args.data_num > 10000: # eval用のデータ数は10,000
            # if epoch * args.batch > 10000: # eval用のデータ数は10,000
                if args.KMMD:
                    print('KMMD:', eval_kmmd.avg)
                if args.FID:
                    out_path = "./outputs/" + args.resume.split('/')[2] + '/'
                    os.mkdir(out_path)
                    for i in range(1000):
                        visualizers.random_eval(model,out_path,tmp=0.3, n=1, truncate=True, roop_n=i)
                break

        if epoch > max_epoch:
            break
        epoch+=1
    
    if args.mode == 'train':
        log_save_path = os.path.join(checkpoint_dir,"train-log.json")
        save_json(log,log_save_path)

if __name__ == '__main__':
    args = argparse_setup()
    main(args)
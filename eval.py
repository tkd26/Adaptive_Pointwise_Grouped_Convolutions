import torch
import torch.nn as nn
import argparse
from loss.KMMD import KMMD_loss
import glob
from dataloaders.ImageListDataset import ImageListDataset
from torchvision import transforms
from torch.utils.data import  DataLoader

def argparse_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)

    parser.add_argument('--dataset', type=str, default="face", help = "dataset. anime or face or flower. ")
    return parser.parse_args()

if __name__ == "__main__":
    args = argparse_setup()
    h, w = [128, 128]
    batch_size = 100
    num_workers = 4

    source_path_list = glob.glob(args.source + '*')[:batch_size]
    target_path_list = glob.glob(args.target + '*')[:batch_size]

    transform = transforms.Compose([
            transforms.Resize( min(h,w) ),
            transforms.CenterCrop( (h,w) ),
            transforms.ToTensor(),
            ])

    source_path_list = [[path,i] for i,path in enumerate(sorted(source_path_list))]
    target_path_list = [[path,i] for i,path in enumerate(sorted(target_path_list))]

    source_dataset = ImageListDataset(source_path_list,transform=transform)
    target_dataset = ImageListDataset(target_path_list,transform=transform)
    
    source_dataloader = DataLoader([data for data in source_dataset],batch_size=batch_size,shuffle=False,num_workers=num_workers)
    target_dataloader = DataLoader([data for data in target_dataset],batch_size=batch_size,shuffle=False,num_workers=num_workers)

    source_data = iter(source_dataloader).next()[0]
    target_data = iter(target_dataloader).next()[0]

    loss = KMMD_loss()(source_data, target_data)
    print(loss)
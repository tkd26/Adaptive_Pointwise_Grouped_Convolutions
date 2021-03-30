import glob
from .ImageListDataset import ImageListDataset
from torchvision import transforms
from torch.utils.data import  DataLoader

def setup_dataloader(name,h=128,w=128,batch_size=4,num_workers=4,data_num=50):
    '''
    instead of setting up dataloader that read raw image from file, 
    let's use store all images on cpu memmory
    because this is for small dataset
    '''
    if name == "face":
        # img_path_list = glob.glob("./data/face/*.png")
        img_path_list = glob.glob("/home/yanai-lab/takeda-m/space/dataset/FFHQ/face500/*.png")
        img_path_list = sorted(img_path_list)[:data_num]
    elif name=="anime":
        img_path_list = glob.glob("/home/yanai-lab/takeda-m/space/dataset/Danbooru2019/anime500/*.jpg")
        img_path_list = sorted(img_path_list)[:data_num]
    elif name=="flower": # 
        img_path_list = glob.glob("/home/yanai-lab/takeda-m/space/dataset/102flowers/passiflora/*.jpg")
        img_path_list = sorted(img_path_list)[:data_num]
    elif name=="bird": # AFRICAN FIREFINCH
        # https://www.kaggle.com/gpiosenka/100-bird-species
        img_path_list = glob.glob("/home/yanai-lab/takeda-m/space/dataset/bird/african_firefinch/*.jpg")
        img_path_list = sorted(img_path_list)[:data_num]
    elif name=="car": # BMW
        # https://ai.stanford.edu/~jkrause/cars/car_dataset.html
        img_path_list = glob.glob("/home/yanai-lab/takeda-m/space/dataset/car/bmw/*.jpg")
        img_path_list = sorted(img_path_list)[:data_num]
    else:
        raise NotImplementedError("Unknown dataset %s"%name)
        
    assert len(img_path_list) > 0

    transform = transforms.Compose([
            transforms.Resize( min(h,w) ),
            transforms.CenterCrop( (h,w) ),
            transforms.ToTensor(),
            ])
    
    img_path_list = [[path,i] for i,path in enumerate(sorted(img_path_list))]
    dataset = ImageListDataset(img_path_list,transform=transform)
    
    return  DataLoader([data for data in  dataset],batch_size=batch_size, 
                            shuffle=True,num_workers=num_workers)
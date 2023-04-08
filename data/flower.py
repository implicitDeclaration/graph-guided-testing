
import torch
import torchvision
from torchvision import transforms



class flower:
    def __init__(self, args):
        super(flower, self).__init__()

        data_dir = '../dataset/flower_data'
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        name_json = data_dir + '/cat_to_name.json'

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data Normalization code 
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

        valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

        image_datasets = {}
        image_datasets['train'] = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
        
        image_datasets['valid'] = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)


        # Using the image datasets and the trainforms, define the dataloaders

        self.train_loader = torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=args.batch_size, shuffle=True, **kwargs
        )


        self.val_loader = torch.utils.data.DataLoader(
            image_datasets['valid'], batch_size=args.batch_size, shuffle=False, **kwargs
        )

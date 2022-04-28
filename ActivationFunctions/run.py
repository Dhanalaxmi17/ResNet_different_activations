import glob, os, shutil
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import json
import random

from sklearn.model_selection import GroupShuffleSplit
import wandb
from datetime import datetime

from models.models_activation_function_main import create_model
from data.chex import CheX_Dataset, ToPILImage, XRayCenterCrop, XRayResizer, SubsetDataset

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

ROOT_DIR = os.path.dirname(__file__)

while True:
    try:
        params = glob.glob(os.path.join("params", "*json"))
        try:
            params_0 = params[0]
            print(os.path.join(ROOT_DIR, params_0))
            cfg = open(os.path.join(ROOT_DIR, params_0))
            print(cfg)
            cfg = json.load(cfg)
            print(cfg)
        except:
            # print("No param files")
            exit()

        # Hyper-parameters
        seed=3407 # try with 0, 37, 3407  #https://arxiv.org/abs/2109.08203
        num_epochs = cfg["training"]["num_epochs"]
        learning_rate = cfg["training"]["learning_rate"]
        batch_size = cfg["dataset"]["batch_size"]
        dataset = cfg["dataset"]["class"]
        model_name = cfg["model"]["class"]
        activation_function = cfg["model"]["activation_function"]
        exp_id = cfg["general"]["id"]
        gpu_id = cfg["general"]["gpu"]["gpu_id"]
        notes_description = cfg["general"]["description"]
        threads = cfg["dataset"]["num_workers"]

        # Device configuration
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

        # Setting the seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # create directory
        current_datetime = datetime.now()
        current_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
        current_datetime = current_datetime[2:]
        os.mkdir("./runs/{}_{}".format(current_datetime, exp_id))


        # Dataset   
        if "chexpert" in dataset.lower(): 
            dataset_dir = cfg["dataset"]["path"]

            transform = transforms.Compose([
                ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees = (-10,10), 
                                        # translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                        scale=(1.0, 1.2)),
                transforms.ToTensor()
            ])

            transfms = torchvision.transforms.Compose([XRayCenterCrop(), XRayResizer(224)])

            dataset_df = CheX_Dataset(
                imgpath= dataset_dir,# + "/CheXpert-v1.0-small",
                csvpath= dataset_dir + "train.csv", #"/CheXpert-v1.0-small/train.csv",
                transform=transfms, data_aug=transform, unique_patients=False)

            gss = GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=seed)
            train_inds, test_inds = next(gss.split(X=range(len(dataset_df)), groups=dataset_df.csv.patientid))
            train_dataset = SubsetDataset(dataset_df, train_inds)
            test_dataset = SubsetDataset(dataset_df, test_inds)

            # Dataloader
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=threads, 
                                                    pin_memory=True
                                                    )
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=threads, 
                                                    pin_memory=True
                                                    )

            classes = ("Enlarged Cardiomediastinum",
                                "Cardiomegaly",
                                "Lung Opacity",
                                "Lung Lesion",
                                "Edema",
                                "Consolidation",
                                "Pneumonia",
                                "Atelectasis",
                                "Pneumothorax",
                                "Pleural Effusion",
                                "Pleural Other",
                                "Fracture",
                                "Support Devices")

            criterion = nn.BCEWithLogitsLoss()

        elif "cifar10" in dataset.lower():
            # CIFAR-10 dataset
            # Image preprocessing modules
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32),
                                            transforms.RandomRotation(10),
                                            transforms.ToTensor()])

            train_dataset = torchvision.datasets.CIFAR10(root="./data/",
                                                        train=True, 
                                                        transform=transform,
                                                        download=True)

            test_dataset = torchvision.datasets.CIFAR10(root="./data/",
                                                        train=False, 
                                                        transform=transforms.ToTensor())

            classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            # Loss
            criterion = nn.CrossEntropyLoss()
            
        elif "mnist" in dataset.lower():
            # MNIST dataset
            # Image preprocessing modules
            transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                            # transforms.RandomCrop(32),
                                            # transforms.RandomRotation(10),
                                            transforms.ToTensor()])

            train_dataset = torchvision.datasets.MNIST(root="./data/",
                                                        train=True, 
                                                        transform=transform,
                                                        download=True)

            test_dataset = torchvision.datasets.MNIST(root="./data/",
                                                        train=False, 
                                                        transform=transforms.ToTensor())

            classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

            # Loss
            criterion = nn.CrossEntropyLoss()
            
        # Model
        model = create_model(model_str = model_name, af_str = activation_function, num_classes = len(classes))
        model.to(device)

        ## LOAD PRETRAINED
        # PATH = './' + model_name + '.ckpt'
        # model.load_state_dict(torch.load(PATH))

        # Optimizer
        # For updating learning rate
        def update_lr(optimizer, lr):    
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True) #, weight_decay=1e-5)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)

        # Train the model
        total_step = len(train_loader)
        curr_lr = learning_rate

        wandbrun = wandb.init(project = "activation_function", allow_val_change = True, reinit=True, notes=notes_description, name=exp_id, entity="ml702")

        for epoch in range(num_epochs):
            ###################### TRAIN ############################
            for i, (images, labels) in enumerate(tqdm(train_loader)):
                images = images.to(device)
                labels = labels.to(device)
                # print(labels.shape)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                wandb.config.update({"Model": model_name,
                                    "Dataset": dataset,
                                    "ActFunc": activation_function})
                wandb.log({"train_loss": loss.item()}, step = int(epoch+1)) #Logging all values into Wandb

            ###################### TEST ############################
            # prepare to count predictions for each class
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}

            # Test the model
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in tqdm(test_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    for label, prediction in zip(labels, predicted):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
                wandb.log({'Accuracy': float(np.round(100 * correct / total, 2))})
                # print accuracy for each class
                for classname, correct_count in correct_pred.items():
                    accuracy = 100 * float(correct_count) / total_pred[classname]
                    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
                                                                accuracy))
                    wandb.log({"acc_{}".format(classname) : float(np.round(accuracy, 2))}, step = int(epoch+2)) #Logging all values into Wandb

            # Decay learning rate
            if (epoch+1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)


        # Save the model checkpoint
        torch.save(model.state_dict(), "./runs/{}_{}/{}".format(current_datetime, exp_id, exp_id + ".ckpt"))

        wandbrun.finish()

        done_path = "./params/done"
        if os.path.exists(done_path):
            shutil.move(params_0, done_path)
        else:
            os.mkdir(done_path)
            shutil.move(params_0, done_path)

    except: 
        failed_path = "./params/failed"
        if os.path.exists(failed_path):
            shutil.move(params_0, failed_path)
        else:
            os.makedirs(failed_path)
            shutil.move(params_0, failed_path)
        print(f'EXPERIMENT {cfg.exp_id} FAILED!!! CONTINUE NEXT.')
        pass

import collections
import os
import os.path
import pprint
import random
import sys
import tarfile
import warnings
import zipfile

import imageio
import numpy as np
import pandas as pd
import skimage
import skimage.transform
from skimage.io import imread
from torchvision import transforms
import torch

default_pathologies = [
    'Atelectasis',
    'Consolidation',
    'Infiltration',
    'Pneumothorax',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Effusion',
    'Pneumonia',
    'Pleural_Thickening',
    'Cardiomegaly',
    'Nodule',
    'Mass',
    'Hernia',
    'Lung Lesion',
    'Fracture',
    'Lung Opacity',
    'Enlarged Cardiomediastinum'
]

thispath = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(thispath, "data")

# this is for caching small things for speed
_cache_dict = {}


def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img


def relabel_dataset(pathologies, dataset, silent=False):
    """
    Reorder, remove, or add (nans) to a dataset's labels.
    Use this to align with the output of a network.
    """
    will_drop = set(dataset.pathologies).difference(pathologies)
    if will_drop != set():
        if not silent:
            print("{} will be dropped".format(will_drop))
    new_labels = []
    dataset.pathologies = list(dataset.pathologies)
    for pathology in pathologies:
        if pathology in dataset.pathologies:
            pathology_idx = dataset.pathologies.index(pathology)
            new_labels.append(dataset.labels[:, pathology_idx])
        else:
            if not silent:
                print("{} doesn't exist. Adding nans instead.".format(pathology))
            values = np.empty(dataset.labels.shape[0])
            values.fill(np.nan)
            new_labels.append(values)
    new_labels = np.asarray(new_labels).T

    dataset.labels = new_labels
    dataset.pathologies = pathologies


class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the 
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)] # Select the view


class MergeDataset(Dataset):
    def __init__(self, datasets, seed=0, label_concat=False):
        super(MergeDataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datasets = datasets
        self.length = 0
        self.pathologies = datasets[0].pathologies
        self.which_dataset = np.zeros(0)
        self.offset = np.zeros(0)
        currentoffset = 0
        for i, dataset in enumerate(datasets):
            self.which_dataset = np.concatenate([self.which_dataset, np.zeros(len(dataset))+i])
            self.length += len(dataset)
            self.offset = np.concatenate([self.offset, np.zeros(len(dataset))+currentoffset])
            currentoffset += len(dataset)
            if dataset.pathologies != self.pathologies:
                raise Exception("incorrect pathology alignment")

        if hasattr(datasets[0], 'labels'):
            self.labels = np.concatenate([d.labels for d in datasets])
        else:
            print("WARN: not adding .labels")

        self.which_dataset = self.which_dataset.astype(int)

        if label_concat:
            new_labels = np.zeros([self.labels.shape[0], self.labels.shape[1]*len(datasets)])*np.nan
            for i, shift in enumerate(self.which_dataset):
                size = self.labels.shape[1]
                new_labels[i, shift*size:shift*size+size] = self.labels[i]
            self.labels = new_labels

        try:
            self.csv = pd.concat([d.csv for d in datasets])
        except:
            print("Could not merge dataframes (.csv not available):", sys.exc_info()[0])

        self.csv = self.csv.reset_index(drop=True)

    def __setattr__(self, name, value):
        if name == "transform":
            raise NotImplementedError("Cannot set transform on a merged dataset. Set the transforms directly on the dataset object. If it was to be set via this merged dataset it would have to modify the internal datasets which could have unexpected side effects")
        if name == "data_aug":
            raise NotImplementedError("Cannot set data_aug on a merged dataset. Set the transforms directly on the dataset object. If it was to be set via this merged dataset it would have to modify the internal datasets which could have unexpected side effects")
            
        object.__setattr__(self, name, value)
        
    
    def string(self):
        s = self.__class__.__name__ + " num_samples={}\n".format(len(self))
        for i, d in enumerate(self.datasets):
            if i < len(self.datasets)-1:
                s += "├{} ".format(i) + d.string().replace("\n", "\n|  ") + "\n"
            else:
                s += "└{} ".format(i) + d.string().replace("\n", "\n   ") + "\n"
        return s

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.datasets[int(self.which_dataset[idx])][idx  - int(self.offset[idx])]
        item["lab"] = self.labels[idx]
        item["source"] = self.which_dataset[idx]
        return item


# alias so it is backwards compatible
Merge_Dataset = MergeDataset


class FilterDataset(Dataset):
    def __init__(self, dataset, labels=None):
        super(FilterDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = []
        if labels:
            for label in labels:
                print("filtering for ", label)

                self.idxs += list(np.where(dataset.labels[:, list(dataset.pathologies).index(label)] == 1)[0])

        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


class SubsetDataset(Dataset):
    def __init__(self, dataset, idxs=None):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = idxs
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]
        self.csv = self.csv.reset_index(drop=True)

        if hasattr(self.dataset, 'which_dataset'):
            self.which_dataset = self.dataset.which_dataset[self.idxs]

    def __setattr__(self, name, value):
        if name == "transform":
            raise NotImplementedError("Cannot set transform on a subset dataset. Set the transforms directly on the dataset object. If it was to be set via this subset dataset it would have to modify the internal dataset which could have unexpected side effects")
        if name == "data_aug":
            raise NotImplementedError("Cannot set data_aug on a subset dataset. Set the transforms directly on the dataset object. If it was to be set via this subset dataset it would have to modify the internal dataset which could have unexpected side effects")
            
        object.__setattr__(self, name, value)
            
    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

class CheX_Dataset(Dataset):
    """CheXpert Dataset
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
    Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, 
    Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong, 
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, 
    Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng. https://arxiv.org/abs/1901.07031
    
    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/
    
    A small validation set is provided with the data as well, but is so tiny, it not included
    here.
    """
    def __init__(self,
                 imgpath,
                 csvpath=os.path.join(datapath, "chexpert_train.csv.gz"),
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True
    ):

        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Enlarged Cardiomediastinum",
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
                            "Support Devices"]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views

        self.csv["view"] = self.csv["Frontal/Lateral"] # Assign view column
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"] # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"}) # Rename Lateral with L

        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat = r'(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # Rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))

        ########## add consistent csv values

        # offset_day_int

        # patientid
        if 'train' in csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif 'valid' in csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplemented

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient","")
        self.csv["patientid"] = patientid

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Path'].iloc[idx]
        imgid = imgid.replace("CheXpert-v1.0-small/","")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        # print(img.shape)
        
        sample["img"] = normalize(img, maxval=255, reshape=True)
        # print(sample["img"].shape)
        # print(type(sample["img"]))

        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])
            
        # MR ADDED
        # print(sample["img"].shape)
        # sample["img"] = torch.moveaxis(sample["img"], 0, -1)
        sample["img"] = sample["img"].expand(3, -1, -1)

        # return sample
        return sample["img"], sample["lab"]

class ToPILImage(object):
    def __init__(self):
        self.to_pil = transforms.ToPILImage(mode="F")

    def __call__(self, x):
        return self.to_pil(x[0])


class XRayResizer(object):
    def __init__(self, size, engine="skimage"):
        self.size = size
        self.engine = engine
        if 'cv2' in sys.modules:
            print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img):
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            import cv2 # pip install opencv-python
            return cv2.resize(img[0,:,:],
                              (self.size, self.size),
                              interpolation = cv2.INTER_AREA
                             ).reshape(1,self.size,self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")


class XRayCenterCrop(object):
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y,x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)


class CovariateDataset(Dataset):
    """Dataset which will correlate the dataset with a specific label.
    
    Viviano et al. Saliency is a Possible Red Herring When Diagnosing Poor Generalization
    https://arxiv.org/abs/1910.00199
    """
    def __init__(self,
                 d1, d1_target,
                 d2, d2_target,
                 ratio=0.5,
                 mode="train",
                 seed=0,
                 nsamples=None,
                 splits=[0.5, 0.25, 0.25],
                 verbose=False
    ):
        super(CovariateDataset, self).__init__()

        self.splits = np.array(splits)
        self.d1 = d1
        self.d1_target = d1_target
        self.d2 = d2
        self.d2_target = d2_target

        assert mode in ['train', 'valid', 'test']
        assert np.sum(self.splits) == 1.0

        np.random.seed(seed)  # Reset the seed so all runs are the same.

        all_imageids = np.concatenate([np.arange(len(self.d1)),
                                       np.arange(len(self.d2))]).astype(int)
        all_idx = np.arange(len(all_imageids)).astype(int)

        all_labels = np.concatenate([d1_target,
                                     d2_target]).astype(int)

        all_site = np.concatenate([np.zeros(len(self.d1)),
                                   np.ones(len(self.d2))]).astype(int)

        idx_sick = all_labels == 1
        n_per_category = np.min([sum(idx_sick[all_site==0]),
                                 sum(idx_sick[all_site==1]),
                                 sum(~idx_sick[all_site==0]),
                                 sum(~idx_sick[all_site==1])])

        if verbose:
            print("n_per_category={}".format(n_per_category))

        all_0_neg = all_idx[np.where((all_site==0) & (all_labels==0))]
        all_0_neg = np.random.choice(all_0_neg, n_per_category, replace=False)
        all_0_pos = all_idx[np.where((all_site==0) & (all_labels==1))]
        all_0_pos = np.random.choice(all_0_pos, n_per_category, replace=False)
        all_1_neg = all_idx[np.where((all_site==1) & (all_labels==0))]
        all_1_neg = np.random.choice(all_1_neg, n_per_category, replace=False)
        all_1_pos = all_idx[np.where((all_site==1) & (all_labels==1))]
        all_1_pos = np.random.choice(all_1_pos, n_per_category, replace=False)

        # TRAIN
        train_0_neg = np.random.choice(
            all_0_neg, int(n_per_category*ratio*splits[0]*2), replace=False)
        train_0_pos = np.random.choice(
            all_0_pos, int(n_per_category*(1-ratio)*splits[0]*2), replace=False)
        train_1_neg = np.random.choice(
            all_1_neg, int(n_per_category*(1-ratio)*splits[0]*2), replace=False)
        train_1_pos = np.random.choice(
            all_1_pos, int(n_per_category*ratio*splits[0]*2), replace=False)

        # REDUCE POST-TRAIN
        all_0_neg = np.setdiff1d(all_0_neg, train_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, train_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, train_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, train_1_pos)

        if verbose:
            print("TRAIN (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                   ratio,
                   len(train_0_neg)+len(train_1_neg),
                   len(train_0_pos)+len(train_1_pos),
                   len(train_0_pos),
                   len(train_0_neg),
                   len(train_1_pos),
                   len(train_1_neg)))

        # VALID
        valid_0_neg = np.random.choice(
            all_0_neg, int(n_per_category*(1-ratio)*splits[1]*2), replace=False)
        valid_0_pos = np.random.choice(
            all_0_pos, int(n_per_category*ratio*splits[1]*2), replace=False)
        valid_1_neg = np.random.choice(
            all_1_neg, int(n_per_category*ratio*splits[1]*2), replace=False)
        valid_1_pos = np.random.choice(
            all_1_pos, int(n_per_category*(1-ratio)*splits[1]*2), replace=False)

        # REDUCE POST-VALID
        all_0_neg = np.setdiff1d(all_0_neg, valid_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, valid_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, valid_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, valid_1_pos)

        if verbose:
            print("VALID (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                   1-ratio,
                   len(valid_0_neg)+len(valid_1_neg),
                   len(valid_0_pos)+len(valid_1_pos),
                   len(valid_0_pos),
                   len(valid_0_neg),
                   len(valid_1_pos),
                   len(valid_1_neg)))

        # TEST
        test_0_neg = all_0_neg
        test_0_pos = all_0_pos
        test_1_neg = all_1_neg
        test_1_pos = all_1_pos

        if verbose:
            print("TEST (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                   1-ratio,
                   len(test_0_neg)+len(test_1_neg),
                   len(test_0_pos)+len(test_1_pos),
                   len(test_0_pos),
                   len(test_0_neg),
                   len(test_1_pos),
                   len(test_1_neg)))


        def _reduce_nsamples(nsamples, a, b, c, d):
            if nsamples:
                a = a[:int(np.floor(nsamples/4))]
                b = b[:int(np.ceil(nsamples/4))]
                c = c[:int(np.ceil(nsamples/4))]
                d = d[:int(np.floor(nsamples/4))]

            return (a, b, c, d)

        if mode == "train":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, train_0_neg, train_0_pos, train_1_neg, train_1_pos)
        elif mode == "valid":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, valid_0_neg, valid_0_pos, valid_1_neg, valid_1_pos)
        elif mode == "test":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, test_0_neg, test_0_pos, test_1_neg, test_1_pos)
        else:
            raise Exception("unknown mode")

        self.select_idx = np.concatenate([a, b, c, d])
        self.imageids = all_imageids[self.select_idx]
        self.pathologies = ["Custom"]
        self.labels = all_labels[self.select_idx].reshape(-1,1)
        self.site = all_site[self.select_idx]

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.imageids)

    def __getitem__(self, idx):

        if self.site[idx] == 0:
            dataset = self.d1
        else:
            dataset = self.d2

        sample = dataset[self.imageids[idx]]

        # Replace the labels with the specific label we focus on
        sample["lab-old"] = sample["lab"]
        sample["lab"] = self.labels[idx]

        sample["site"] = self.site[idx]

        return sample
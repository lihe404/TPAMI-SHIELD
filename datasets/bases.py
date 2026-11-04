from PIL import Image, ImageFile
import json
from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import logging
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        logger.info("  ----------------------------------------")

class ImageDataset(Dataset):
    def __init__(self, cfg, dataset_type, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.dataset_type = dataset_type

        self.description = None
        if cfg.DATASETS.NAMES == 'market1501':
            if self.dataset_type == 'train':
                self.latent_dict = torch.load("/data1/mqx_log/latent_market_bth1with_ins_1notrans_dict.pt")
                # with open("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/description/img_description/all_market_person_descriptions.json", 'r') as f:
                #   self.description = json.load(f)
            else:
                self.latent_dict = torch.load("/data1/mqx_log/latent_market_bth1with_ins_1query_gallery_dict.pt")
                # with open("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/description/img_description/all_val_market_person_descriptions.json", 'r') as f:
                #   self.description = json.load(f)
        elif cfg.DATASETS.NAMES == 'msmt17':
            if self.dataset_type == 'train':
                self.latent_dict = torch.load("/data1/mqx_log/latent_msmt_bth1with_ins_1notrans_dict.pt")
                # with open("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/description/img_description/all_msmt_person_descriptions.json", 'r') as f:
                #   self.description = json.load(f)
            else:
                self.latent_dict = torch.load("/data1/mqx_log/latent_msmt_bth1with_ins_1query_gallery_dict.pt")
                # with open("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/description/img_description/all_val_msmt_person_descriptions.json", 'r') as f:
                #   self.description = json.load(f)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        # print(self.latent_dict)
        # ————————————————————————————————————————————————
    
        latent = self.latent_dict[img_path]
        # if self.description is not None:
        #     description = self.description[img_path]
        # else: 
        description = ''

        # latent = torch.tensor(0)

        # description = ''
        # print(latent)
        # ———————————————————————————————————————————————— 
        if self.transform is not None:
            img = self.transform(img)


        return img, pid, camid, trackid, img_path, latent, description

        #  return img, pid, camid, trackid,img_path.split('/')[-1]

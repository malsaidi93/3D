import torch
from torch.utils.data import Dataset, DataLoader
import os, math
import sys
sys.path.append('../')
import json
import time
import random
import numpy as np
import cv2
import glob
from pathlib import Path
from uois3D.src import data_augmentation 
from uois3D.src.util import utilities as util_
# import open3d as o3d



###### Some utilities #####

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)



class OCIDObject(Dataset):
#     def __init__(self, split,config):
    def __init__(self, split):

        

        self._name = 'ocid_data_' + split
#         self.config = config
        self._split = split

        self._ocid_object_path = '/opt/working/3D/3d-project/data/external/OCID-dataset'
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        self._width = 640
        self._height = 480
        self.image_paths = self.list_dataset()
#         self.image_paths_train = self.list_dataset_train()
#         self.image_paths_test = self.list_dataset_test()
        
        print('%d images for dataset %s' % (len(self.image_paths), self._name))
        self._size = len(self.image_paths)
        assert os.path.exists(self._ocid_object_path), \
                'ocid_object path does not exist: {}'.format(self._ocid_object_path)
        


    def list_dataset(self):
        
        
        if self._split == 'train':
            data_path = '/opt/working/3D/3d-project/data/external/train_ocid.txt'
            
        elif self._split == 'test':
            data_path = '/opt/working/3D/3d-project/data/external/test_ocid.txt'
        else:
            data_path = '/opt/working/3D/3d-project/data/external/all_ocid.txt'
#         seqs = list(Path(data_path).glob('**/*seq*'))
        with open(data_path) as f:
            content = f.readlines()
        image_paths = [content[i].strip('\n') for i in range(len(content))]  
        

#         image_paths = []
#         for seq in seqs:
#             paths = sorted(list((seq / 'rgb').glob('*.png')))
#             image_paths += paths
        return image_paths
    


    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels
    
    def process_rgb(self,img):
        """ Process RGB image
        """
        rgb_img = img.astype(np.float32)

        rgb_img = data_augmentation.standardize_image(rgb_img)

        return rgb_img


    def __getitem__(self, idx):
#         Camera Parameters 
        
        camera_params_path = '/opt/working/3D/3d-project/uois_/src/camera_params.json'
        with open(camera_params_path) as json_file:
            camera_params = json.load(json_file)
        
#         RGB Images

        filename = str(self.image_paths[idx])
    
        rgb_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        rgb_img = self.process_rgb(rgb_img)     

        # Label
        
        labels_filename = filename.replace('rgb', 'label')
        foreground_labels = util_.imread_indexed(labels_filename)
        # mask table as background
        foreground_labels[foreground_labels == 1] = 0
        if 'table' in labels_filename:
            foreground_labels[foreground_labels == 2] = 0
        foreground_labels = self.process_label(foreground_labels)
#         labels = foreground_labels

        index = filename.find('OCID')

        
#         Depth
        depth_filename = filename.replace('rgb', 'depth')
        depth_img = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
        depth=(depth_img/1000).astype(np.float32)
        xyz_img = util_.compute_xyz(depth,camera_params)
#         if self.config['use_data_augmentation']:
#             xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth, self.config)
        
        
#         rgb_img = data_augmentation.array_to_tensor(rgb_img) # Shape: [3 x H x W]
#         xyz_img = data_augmentation.array_to_tensor(xyz_img) # Shape: [3 x H x W]
#         foreground_labels = data_augmentation.array_to_tensor(foreground_labels) # Shape: [H x W]
        
        
#         sample['depth'] = depth
# 'filename': filename[index+5:],
#         sample['xyz'] = xyz_img
        
        sample = {'rgb': rgb_img,
                  'label': foreground_labels,
                  'xyz': xyz_img
                 }

        return sample
    
    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where ocid_object is expected to be installed.
        """
        return '/opt/working/3D/3d-project/data/external/OCID-dataset'

    
def train_dataloader(data_dir, config, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = OCIDObject('train',config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)



def test_dataloader(data_dir, config, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = OCIDObject('test')

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)
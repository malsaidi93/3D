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
from uois2D.src import data_augmentation 
from uois2D.src.util import utilities as util_
# import open3d as o3d



###### Some utilities #####

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

    
    
    
def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.

        If focal lengths fx,fy are stored in the camera_params dictionary, use that.
        Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 
    """

    # Compute focal length from camera parameters
    if 'fx' in camera_params and 'fy' in camera_params:
        fx = camera_params['fx']
        fy = camera_params['fy']
    else: # simulated data
        aspect_ratio = camera_params['img_width'] / camera_params['img_height']
        e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
        t = camera_params['near'] / e; b = -t
        r = t * aspect_ratio; l = -r
        alpha = camera_params['img_width'] / (r-l) # pixels per meter
        focal_length = camera_params['near'] * alpha # focal length of virtual camera (frustum camera)
        fx = focal_length; fy = focal_length

    if 'x_offset' in camera_params and 'y_offset' in camera_params:
        x_offset = camera_params['x_offset']
        y_offset = camera_params['y_offset']
    else: # simulated data
        x_offset = camera_params['img_width']/2
        y_offset = camera_params['img_height']/2

    indices = util_.build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    
    return xyz_img
    
    

class OCIDObject(Dataset):

    def __init__(self, split,config):

        

        self._name = 'ocid_data_' + split
        self.config = config
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
            data_path = '../data/external/train_ocid.txt'
            
        elif self._split == 'test':
            data_path = '../data/external/test_ocid.txt'
        else:
            data_path = '../data/external/test_ocid.txt'
#         seqs = list(Path(data_path).glob('**/*seq*'))
        with open(data_path) as f:
            content = f.readlines()
        image_paths = [content[i].strip('\n') for i in range(len(content))]  

        return image_paths

    
    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
                     direction_labels: a [H x W x 2] numpy array of 2D directions. The i,j^th element has (y,x) direction to object center
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels

        # Compute object centers and directions
        H, W = foreground_labels.shape
        direction_labels = np.stack([np.ones((H,W)), np.zeros((H, W))], axis=-1).astype(np.float32) # Shape: [H x W x 2]
        pixel_indices = util_.build_matrix_of_indices(H, W)
        for k in np.unique(foreground_labels):

            if k in [0, 1]: # background, table
                continue

            # Get object mask
            object_mask = foreground_labels == k

            # Get average of all pixel indices in mask
            center = np.mean(pixel_indices[object_mask, :], axis=0) # Shape: [2]. y_center, x_center

            # Get directions
            object_center_directions = (center - pixel_indices).astype(np.float32) # Shape: [H x W x 2]
            object_center_directions = object_center_directions / np.maximum(np.linalg.norm(object_center_directions, axis=2, keepdims=True), 1e-10)

            # Add it to the labels
            direction_labels[object_mask] = object_center_directions[object_mask]

        return foreground_labels, direction_labels
    
    def process_rgb(self,img):
        """ Process RGB image
        """
        rgb_img = img.astype(np.float32)

        rgb_img = data_augmentation.standardize_image(rgb_img)

        return rgb_img


    def process_depth(self, depth_img, seg_img):
        """ Process depth channel
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # add random noise to depth
        if self.params['use_data_augmentation']:
            depth_img = data_augmentation.add_noise_to_depth(depth_img, self.params)
            depth_img = data_augmentation.dropout_random_ellipses(depth_img, self.params)

        # Compute xyz ordered point cloud and add noise
        xyz_img = compute_xyz(depth_img, self.params)
        if self.params['use_data_augmentation']:
            xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth_img, self.params)

        return xyz_img
    
    def __getitem__(self, idx):
#         Camera Parameters 
        
        camera_params_path = '/opt/working/3D/3d-project/uois3D/src/camera_params.json'
        with open(camera_params_path) as json_file:
            camera_params = json.load(json_file)
        
#         RGB Images

        filename = str(self.image_paths[idx])
    
        rgb_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        rgb_img = self.process_rgb(rgb_img)     

        # Label
        
        labels_filename = filename.replace('rgb', 'label')
        foreground_labels = util_.imread_indexed(labels_filename)
        foreground_labels, direction_labels = self.process_label(foreground_labels)
#         index = filename.find('OCID')

        
#         Depth
        depth_filename = filename.replace('rgb', 'depth')
        depth_img = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
        depth = (depth_img / 1000.).astype(np.float32)
       
        xyz_img = compute_xyz(depth,camera_params)
        
        
        
        rgb_img = data_augmentation.array_to_tensor(rgb_img) # Shape: [3 x H x W]
        xyz_img = data_augmentation.array_to_tensor(xyz_img) # Shape: [3 x H x W]
        
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels) # Shape: [H x W]
        direction_labels = data_augmentation.array_to_tensor(direction_labels) # Shape: [2 x H x W]
        

        
        sample = {'rgb': rgb_img,
                  'foreground_labels': foreground_labels,
                  'direction_labels' : direction_labels,
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

    
def get_train_dataloader(data_dir, config, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = OCIDObject('train',config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)



def get_test_dataloader(data_dir, config, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = OCIDObject('test',config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)
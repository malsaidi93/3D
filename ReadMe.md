# 3D Project 

This repository provides an implementation of 3D instance segmentation model as a first step in exploring 3D space for our projects.

# Repository structure 
```bash
.
├── LICENSE
├── Makefile
├── ReadMe.md
├── data
│   ├── external
│   │   ├── Area5_data
│   │   ├── NYU-V2
│   │   ├── OCID-dataset
│   │   ├── TOD
│   │   ├── all_ocid.txt
│   │   ├── s3dis
│   │   ├── test_ocid.txt
│   │   └── train_ocid.txt
│   ├── interim
│   ├── processed
│   └── raw
│       ├── OCID-dataset.tar.gz
│       ├── TOD.rar
│       ├── data_area5a.zip
│       └── s3dis_h5_07042019.zip
├── docs
│   ├── Makefile
│   ├── commands.rst
│   ├── conf.py
│   ├── getting-started.rst
│   ├── index.rst
│   └── make.bat
├── models
├── notebooks
│   ├── extract_nyu.ipynb
│   ├── extract_s3dis_h5.ipynb
│   ├── sample_s3dis_data.ipynb
│   ├── test_ocid.ipynb
│   ├── uois2D_train_DSN.ipynb
│   ├── uois_inference_2D.ipynb
│   └── uois_inference_3D.ipynb
├── pics
│   ├── DSN_Output_uois2d.PNG
│   ├── inference2D.PNG
│   ├── inference3D.PNG
│   └── uois2d_ocid_training1.PNG
├── references
├── reports
│   └── figures
├── requirements.txt
├── setup.py
├── src_
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   ├── ocid_data_.cpython-37.pyc
│   │   ├── ocid_data_2D.cpython-37.pyc
│   │   └── ocid_data_inference.cpython-37.pyc
│   ├── data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── predict_model.py
│   │   └── train_model.py
│   ├── ocid_data_.py
│   ├── ocid_data_2D.py
│   ├── ocid_data_inference.py
│   └── visualization
│       ├── __init__.py
│       └── visualize.py
├── test_environment.py
├── tox.ini
├── tree11_dir.txt
├── tree1_dir.txt
├── treeFf.txt
├── tree_dir.txt
├── uois2D
│   ├── DSN_Output.PNG
│   ├── DSN_Output_.PNG
│   ├── DSN_Output_uois2d.PNG
│   ├── LICENSE
│   ├── README.md
│   ├── eigen-git-mirror
│   │   ├── CMakeLists.txt
│   │   ├── COPYING.BSD
│   │   ├── COPYING.GPL
│   │   ├── COPYING.LGPL
│   │   ├── COPYING.MINPACK
│   │   ├── COPYING.MPL2
│   │   ├── COPYING.README
│   │   ├── CTestConfig.cmake
│   │   ├── CTestCustom.cmake.in
│   │   ├── Eigen
│   │   ├── INSTALL
│   │   ├── README.md
│   │   ├── bench
│   │   ├── blas
│   │   ├── build
│   │   ├── cmake
│   │   ├── debug
│   │   ├── demos
│   │   ├── doc
│   │   ├── eigen3.pc.in
│   │   ├── failtest
│   │   ├── lapack
│   │   ├── scripts
│   │   ├── signature_of_eigen3_matrix_library
│   │   ├── test
│   │   └── unsupported
│   ├── env.yml
│   ├── example_images
│   ├── gifs
│   │   ├── refinement.gif
│   │   └── robot_grasping.gif
│   ├── src
│   │   ├── __pycache__
│   │   ├── data_augmentation.py
│   │   ├── data_loader.py
│   │   ├── evaluation.py
│   │   ├── hough_voting
│   │   ├── losses.py
│   │   ├── networks.py
│   │   ├── segmentation.py
│   │   └── util
│   ├── tr.png
│   ├── train_DSN.ipynb
│   ├── train_RRN.ipynb
│   ├── uois2D_checkpoints
│   │   ├── DepthSeedingNetwork_TOD_checkpoint.pth
│   │   ├── RRN_OID_checkpoint.pth
│   │   └── RRN_TOD_checkpoint.pth
│   ├── uois2d_ocid_training1.png
│   └── uois_example.ipynb
└── uois3D
    ├── LICENSE
    ├── README.md
    ├── checkpoints
    │   ├── DepthSeedingNetwork_3D_TOD_checkpoint.pth
    │   ├── RRN_OID_checkpoint.pth
    │   └── RRN_TOD_checkpoint.pth
    ├── env.yml
    ├── example_images
    │   ├── OCID_image_0.npy
    │   ├── OCID_image_1.npy
    │   ├── OSD_image_0.npy
    │   └── OSD_image_1.npy
    ├── gifs
    │   └── pipeline.gif
    ├── src
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── camera_params.json
    │   ├── cluster.py
    │   ├── data_augmentation.py
    │   ├── data_loader.py
    │   ├── evaluation.py
    │   ├── losses.py
    │   ├── networks.py
    │   ├── ocid_data_.py
    │   ├── ocid_data_inference.py
    │   ├── segmentation.py
    │   ├── train.py
    │   └── util
    ├── tb_dir_TOD
    │   └── train_DSN
    ├── tb_train_ocid
    │   └── train_DSN
    ├── train_DSN.ipynb
    ├── train_RRN.ipynb
    ├── uois_3D_example.ipynb
    └── uois_inference.ipynb

56 directories, 128 files


```
# Documents

## [_Project Proposal_](https://teams.microsoft.com/l/file/CE1D4392-4383-4BC4-8E4A-B20BA66A452D?tenantId=3b07dc1f-22e7-4be1-ac66-a88bf3550222&fileType=docx&objectUrl=https%3A%2F%2Fverisk.sharepoint.com%2Fsites%2FCIADataScience%2FShared%20Documents%2FCIA%20DL%20Team%2F3D%20Discovery%2FProject%20Proposal_v2.docx&baseUrl=https%3A%2F%2Fverisk.sharepoint.com%2Fsites%2FCIADataScience&serviceName=teams&threadId=19:22ad954b1da34743a9f2b28105fa3bad@thread.tacv2&groupId=620eb361-fbd4-401f-8a34-b7c5586ebe7d)

Document containing the project propsal. 

## [_DataDiscovery_](https://teams.microsoft.com/l/file/2B3FCE4C-DD37-412F-AD2E-43AED15061E7?tenantId=3b07dc1f-22e7-4be1-ac66-a88bf3550222&fileType=docx&objectUrl=https%3A%2F%2Fverisk.sharepoint.com%2Fsites%2FCIADataScience%2FShared%20Documents%2FCIA%20DL%20Team%2F3D%20Discovery%2FDataDiscoveryReport.docx&baseUrl=https%3A%2F%2Fverisk.sharepoint.com%2Fsites%2FCIADataScience&serviceName=teams&threadId=19:22ad954b1da34743a9f2b28105fa3bad@thread.tacv2&groupId=620eb361-fbd4-401f-8a34-b7c5586ebe7d)

This report offers a comprehensive and full overview on 3D data representations in the vision domain. Our focus will be on three representations (RGB-D, Point Cloud) while researching models for implementing our tasks. At this stage we are confident that RGB-D offers a simple yet powerful representation for 3D vision. It allows for transformations on the data while holding the same structure. The flexibility of moving across 3D representations allows us to consider various modeling approaches independent of the input data representation. The availability of RGB-D data paired with camera metadata could serve as an extension to Point Clouds and Voxel representations.  

## [_Model Discovery_](https://teams.microsoft.com/l/file/5E77575A-454E-4A5F-AD18-729BF6A9F32B?tenantId=3b07dc1f-22e7-4be1-ac66-a88bf3550222&fileType=docx&objectUrl=https%3A%2F%2Fverisk.sharepoint.com%2Fsites%2FCIADataScience%2FShared%20Documents%2FCIA%20DL%20Team%2F3D%20Discovery%2FModelDiscoveryReport.docx&baseUrl=https%3A%2F%2Fverisk.sharepoint.com%2Fsites%2FCIADataScience&serviceName=teams&threadId=19:22ad954b1da34743a9f2b28105fa3bad@thread.tacv2&groupId=620eb361-fbd4-401f-8a34-b7c5586ebe7d)

This report is aimed to present a high-level overview of 3D instance segmentation on data representations encompassing RGB-D, voxels and point clouds. The model discovery section yielded a shortlist of models that are considered for our task. The models range were released in the last 3 years and have achieved state of the art results on benchmark datasets such as ScanNetv2 and S3DIS.


<table>
  <tr>
    <td>Models</td>
    <td>Pros</td>
    <td>Cons</td>
  </tr>
  <tr>
    <td>UOIS</td>
    <td>
    <div style="word-wrap: break-word;">
       RGB-D data and generates point cloud <BR>Light weight model with two main networks   
    </div>
        </td>
    <td>
    <div style="word-wrap: break-word;">
       Backbone models in both networks does not use Minkowski networks <BR> Models was trained and evaluated on tabletop environment datasets (OCID-OSD) 
    </div>
    </td>
    </tr>
 <tr>
    <td>3D-SIS</td>
    <td>
    <div style="word-wrap: break-word;">
       incorporates RGB images with voxelized geometry <BR> projects features extracted from 2D views into 3D space
    </div>
        </td>
    <td>
    <div style="word-wrap: break-word;">
       Model is computationally heavy<BR> for training computational resources are wasted on void space
    </div>
    </td>
    </tr>
    
    
 <tr>
    <td>PointGroup</td>
    <td>
    <div style="word-wrap: break-word;">
       2nd highest score on ScanNetv2 <BR> Fast and efficient model <BR> sUses Sparse ConvNets  
    </div>
        </td>
    <td>
    <div style="word-wrap: break-word;">
       Does not incorporate RGB feature extraction in model<BR> Sensitive to hyperparameters that require manual setup for fine tuning 
    </div>
    </td>
    </tr>
 <tr>
    <td>DYCo3D</td>
    <td>
    <div style="word-wrap: break-word;">
       Highest score on ScanNetv2 (2021) <BR> Fast, Robust and efficient model
    </div>
        </td>
    <td>
    <div style="word-wrap: break-word;">
       Does not incorporate RGB feature extraction in model
    </div>
    </td>
    </tr>
    
</table>

    
# Datasets 

Datasets explored in this repository are:
1.  [_NYU-v2_](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) |--> Labeled dataset containing ~1449 RGB-D pairs as well as annotated gt. 
2.  [_OCID_](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/) |--> Labeled dataset contating ~2300 RGB-D pairs, pcd files and annotated gt.
3.  [_S3DIS_](http://buildingparser.stanford.edu/dataset.html) |Using Area5 (downloaed both with XYZ and noXYZ) datasets from the main website | ~2.3 GB of preprocessed data from [JSIS3D](https://github.com/pqhieu/jsis3d)
4.  [_TOD_](https://drive.google.com/uc?export=download&id=1Du309Ye8J7v2c4fFGuyPGjf-C3-623vw) |--> Dataset was provided by the authors of (UOIS3D) and it contains ~34GB of annotated synthetic RGB-D pairs.

    
**Data Preparation Notebooks in ./notebooks/**

| Notebook         | Description|
| :---:            |    :----:  |
| extract_nyu      | Notebook to extract downloaded NYU-v2 labeled dataste (.mat) extension. |       
| test_coid        | Test OCIDObject class in ocid_data_ script to load data from OCID dataset| 
| extract_s3dis_h5 | Extract s3dis h5 files. Dataset downloaded form jsis3d repository (referenced) in datasets|
| sample_s3dis_data| Sample data from extracted s3dis/area5a dataset. |
    

    
# Model Used 

The model implemented in this repository is [_UOIS3D_](https://github.com/chrisdxie/uois) model. The model repository cloned as a submodule in our working direcotry. 

## Notes on UOIS(3D/2D)

1.  UOIS github repository has two main branches which are master and uois2D
2.  Our focus was to use the master branch (UOIS3D) which is it is an extension to uois2D
3.  Both models were trained and benchmarked on the same datasets. UOIS3D acheives slightly better results.
4.  UOIS3D is desired for the ability to identify and segment occluded objects as opposed to uois2d.
5.  Refer to [Inference-Test] for information on inference tests ran on both models.
    
## Clone Models

    git clone https://github.com/chrisdxie/uois.git --> UOIS3D
    
    git clone --branch uois2D https://github.com/chrisdxie/uois.git --> UOIS2D

## Inference-Test

UOIS model was trained on synthetic RGB-D data and was benchmarked on OCID  and OSID datasets. 
**Inference test**

1. Dir ./notebooks/uois3D_inference.ipynb **OR** ./notebooks/uois2D_inference.ipynb 
2. Dataset used 'OCID' --> Located in ./data/external/OCID-dataset
3. Data loader --> ./uois_/src/ocid_data_inference.py 
4. Parameters defined in the notebook (dsn_config, rrn_config, uois_config) were copied from uois_3D_example.ipynb (model directory)
    - Using .__getitem__ method to retrieve 1 sample of data 
    - Implemented in for loop to generate a batch of size N
    - Random sampling from dataset
    
4. Inference test result 
    - 3D Inference
    - ![3D Inference](pics/inference3D.PNG)
    - 2D Inference
    ![2D Inference](pics/inference2D.PNG)
   
5. Notes on Inference Test:
    
    - Example images provided in (../uois_/example_images) are preprocesssed and not raw files pulled directly from the dataset. 
    - Data for inference was sampled from OCID dataset and are similarly preprocessed.
    - Data loading script (src/ocid_data_inference.py) for preprocessing OCID data added to sample and process OCID data. 
    - Data loading scripts (src/ocid_data_inference and src/ocid_data_) have the same implementation for preprocessing and sampling data.<BR> The difference is ocid_data_inference does not require a _config_ dict of parameters to be constrcuted. 
    

# Traning
    Initially the goal was to train the model on S3DIS dataset as it contains scene scans of areas (office,room,..) 
    and the dataset is comprehensive and includes all the data we would want. 

    1.  UOIS3D
    
        - Prior to spending time preparing the data we attempted an initial run of training for 1 epoch on TOD dataset (same dataset model was trained on) using the same files and data               loading scripts provided in the model repository. 
        - Training runs were not successful on UOIS3D model because of what appears to be missing files from TOD dataset. 
        - The root of the issue is:
            - process_label_3D(self, foreground_labels, xyz_img, scene_description) *returns* center_offset_labels, object_centers
            - scene_description which should be present in the dataset as scene_description.txt was missing 
    2.  UOIS2D
    
        - Following the issue presented above we looked into uois2d branch that process labels in 2D settings and computes object_centers using hough_voting algorithm bypassing the                   scene_description requirment.
        - We attempted a training on TOD dataset for 1 epoch and the training was successful. 
        
        - Training on OCID Dataset
            - train_DSN.ipynb notebook was used to train DSN network on OCID dataset for 1 epoch 
            - data loading script was slighty modified and is found in src_/ocid_data_2D.py
            - Main issue preventing us from training using UOIS3D were finding center_offset_labels and object_centers (process_label_3D)
            - This was alleviated in UOIS2D model, direction to centers is an output of process_label and object_centers were estimated and found using hough_voting algorithm
    
    3. Training DSN
        - DSN Network was trained using train_DSN.ipynb in main model repositroy 
        - We used OCID dataset for training 
        - Loss plots
![Loss Plots](pics/uois2d_ocid_training1.PNG)
    
        - Visualizing DSN Output:
            - We expect DSN to output Initial Mask that will be later processed by Mask Processing unit and refined by RRN
![DSN Output](pics/DSN_Output_uois2d.PNG)
    
    4. Training RRN
        - RRN training was not successful at first run. 
    
# Considerations for Moving Forward
   **Model**
    - UOIS model is promising for our use case as it uses RGB-D data with relatively low computational cost and data processing
    - Bugs faced with UOIS3D (missing scene_description file) causes an issue for training, solved with using UOIS2D model which
        - Published paper related to the model states that uois3d has better separation of objects in occluded scenes 
   **Data**
    - Data considered for training is s3dis dataset.
    - A small subset is made available and can be found on [JSIS3D](https://github.com/pqhieu/jsis3d) but the data is preprocessed and utilizing it depends on the model used 
    - Notebooks related to data are found in the notebooks directory and are useful for data extraction and preparation
    
    
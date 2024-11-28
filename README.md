# lidar-camera-road-autolabeling
[[`Video`](https://www.youtube.com/watch?v=sMpXFS5VvJk)] [[`Arxiv`](https://www.youtube.com/watch?v=sMpXFS5VvJk)]

Source code for the paper: **"Trajectory-based road auto labeling with lidar-camera fusion in Winter Conditions"**

Eerik Alamikkotervo, Henrik Toikka, Kari Tammi, Risto Ojala

<img src="https://github.com/user-attachments/assets/ed5e83c8-ffc0-4c2b-862c-72ec9df6e12a" alt="Image description" height="400"/>


## Installation

```
git clone git@github.com:eerik98/lidar-camera-road-autolabeling.git
cd lidar-camera-road-autolabeling
conda env create -f conda.yaml
conda activate lidar-camera-road-autolabeling
```

## Usage

For now, we provide scripts for testing with CADCD and KITTI360. The parameters for CADCD are provided in `params_cadcd.yaml` and for KITTI360 in `params_kitti360.yaml`. **Remember to change the dataset_path in the parameter file**.

If you wish to use your own data, provide it in the same format as our sampling script output and then do steps 3-6. You should define suitable parameters. For 32-channel lidar, `params_cadcd_yaml` is a good starting point, and for 64-channel lidar, `kitti360.yaml` is a good starting point. Finally, you need to define the calibration between your sensors. **We recommend trying out with the provided datasets before moving on to your own data**.  

### 1. Download data
#### CADCD: 
Download the CADCD raw dataset.[`Homepage`](http://cadcd.uwaterloo.ca) [`Devkit`](https://github.com/mpitropov/cadc_devkit). Our methods requires downloading image_00, novatel_rtk, lidar_points and calib.

When using the download script provided in the Devkit the folder structure doesn't need to be modified. 

#### KITTI-360:
Download the KITTI360 dataset. [`Homepage`](https://www.cvlibs.net/datasets/kitti-360/). Our method requires downloading: Calibrations, OXTS sync measurements, Raw Velodyne scans, Left Perspective images (image_00), and Semantics of Left Perspective camera (For validation purposes).   

Place all the data folders in a folder named raw. The path to the images should then be `<dataset_path>/raw/data_2d_raw`, to scans `<dataset_path>/raw/data_3d_raw`, to OXTS `<dataset_path>/raw/data_poses_oxts`, calibrations `<dataset_path>/raw/calibration` and semantics `<dataset_path>/raw/data_2d_semantics`


### 2. Sample

Run the sampling script.
```
python3 sample_cadcd.py <sequence> <parameter file> #for sampling cadcd
python3 sample_cadcd.py '2019_02_27/0006' 'params_cadcd.yaml' # cadcd example
python3 sample_kitti360.py <sequence> <parameter file> #for sampling kitti360
python3 sample_kitti360.py '2013_05_28_drive_0007' 'params_kitti360.yaml' # kitti360 example
```
The sampled data is saved under `<dataset_path>/processed/<sequence>/data`.


### 3. Pre-process
Run the pre-processing script:
```
python3 pre_processing.py <sequence> <parameter file>
python3 pre_processing.py '2019_02_27/0006' 'params_cadcd.yaml' # cadcd example
python3 pre_processing.py '2013_05_28_drive_0007' 'params_kitti360.yaml' #kitti360 example
```
The outputs are saved under `<dataset_path>/processed/<sequence>/autolabels/pre_processing`.

<img src="https://github.com/user-attachments/assets/ebcf04b3-52dd-4965-9d07-72a9048e9fe7" alt="Image description" height="100"/>
<img src="https://github.com/user-attachments/assets/6d64fb59-d2d0-42b4-bfcb-7477bb59913c" alt="Image description" height="100"/>


### 4. Lidar autolabel
Run the lidar-autolabeling script:
```
python3 lidar_autolabeling.py <sequence> <parameter file>
python3 lidar_autolabeling.py '2019_02_27/0006' 'params_cadcd.yaml' # cadcd example
python3 lidar_autolabeling.py '2013_05_28_drive_0007' 'params_kitti360.yaml' #kitti360 example
```
The outputs are saved under `<dataset_path>/processed/<sequence>/autolabels/lidar`.

<img src="https://github.com/user-attachments/assets/3f3106c0-7a51-4e14-8c02-c688fae93666" alt="Image description" height="100"/>
<img src="https://github.com/user-attachments/assets/7ce016b8-2cd5-4afb-ab81-62adbca74681" alt="Image description" height="100"/>

### 5. Camera autolabel
Run the visual autolabeling script:
```
python3 camera_autolabeling.py <sequence> <parameter file>
python3 camera_autolabeling.py '2019_02_27/0006' 'params_cadcd.yaml' # cadcd example
python3 camera_autolabeling.py '2013_05_28_drive_0007' 'params_kitti360.yaml' #kitti360 example
```
The outputs are saved under `<dataset_path>/processed/<sequence>/autolabels/camera`.

<img src="https://github.com/user-attachments/assets/228397e2-43b4-4cae-8c9c-18cecc463585" alt="Image description" height="100"/>
<img src="https://github.com/user-attachments/assets/4ea72fe8-9cfd-4959-ae0a-c3c86ad030d0" alt="Image description" height="100"/>

### 6. Post-process
Run the post-processing script:
```
python3 post_processing.py <sequence> <parameter file>
python3 post_processing.py '2019_02_27/0006' 'params_cadcd.yaml' # cadcd example
python3 post_processing.py '2013_05_28_drive_0007' 'params_kitti360.yaml' #kitti360 example
```
The outputs are saved under `<dataset_path>/processed/<sequence>/autolabels/post_processing`.

<img src="https://github.com/user-attachments/assets/60d7008a-81b4-4e3c-80e9-9060d93319b0" alt="Image description" height="100"/> 
<img src="https://github.com/user-attachments/assets/921defd3-f545-4092-9d22-da6acfd71821" alt="Image description" height="100"/> 

### 7. Evaluate (Only available for KITTI-360)
Run the evaluation script:
```
python3 evaluate_kitti360.py <sequence> <parameter file>
python3 evaluate_kitti360.py '2013_05_28_drive_0007' 'params_kitti360.yaml' #kitti360 example
```
Performance metrics are printed to the terminal. 

## Citing

If you use our code or the related paper in your research, please consider citing.

```bibtex
```



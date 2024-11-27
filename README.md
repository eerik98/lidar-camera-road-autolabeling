# lidar-camera-road-autolabeling

Source code for the paper: "Trajectory-based road auto labeling with lidar-camera fusion"

## Video Demo
[![Video](http://img.youtube.com/vi/sMpXFS5VvJk/0.jpg)](https://www.youtube.com/watch?v=sMpXFS5VvJk)

## Installation

```
git clone git@github.com:eerik98/lidar-camera-road-autolabeling.git
cd lidar-camera-road-autolabeling
conda env create -f conda.yaml
conda activate lidar-camera-road-autolabeling
```

## Usage

For now, we provide scripts for testing with CADCD and KITTI360. The parameters for CADCD are provided in `params_cadcd.yaml` and for KITTI360 in `params_kitti360.yaml`. **Remember to change the dataset_path in the parameter file**.

If you wish to use your own data, provide it in the same format as our sampling script output and then do steps 3-6. Additionally, you should define suitable parameters. For 32-channel lidar, `params_cadcd_yaml` is a good starting point, and for 64-channel lidar, `kitti360.yaml` is a good starting point. 

### 1. Download data
#### CADCD: 
Download the CADCD raw dataset.[`Homepage`](http://cadcd.uwaterloo.ca) [`Devkit`](https://github.com/mpitropov/cadc_devkit)

In `cadcd/raw/<day>/calib/extrinsics.yaml` `T_LIDAR_GPSIMU` change the last column of the second row from `0.0726...` to `-1.5`. Otherwise, the trajectory will be misaligned. We believe there is an error in the provided calibration matrix. 

When using the download script provided in the Devkit the folder structure doesn't need to be modified. 

#### KITTI-360:
Download the KITTI360 dataset. [`Homepage`](https://www.cvlibs.net/datasets/kitti-360/). Our method requires downloading: Calibrations, OXTS sync measurements (GNSS), Raw Velodyne scans, and Perspective images. 
Place the data in a folder named raw. The path to the images should then be `<dataset_path>/raw/data_2d_raw`, to scans `<dataset_path>/raw/data_3d_raw`, to OXTS `<dataset_path>/raw/data_poses_oxts` and calibrations `<dataset_path>/raw/calibration`.  


### 2. Sample

Run the sampling script: 
```
python3 sample_cadcd.py <sequence> <parameter file>
```
`<sequence>` defines the sequence to be processed. For cadcd it is of format `<day>/<sequence_number>`. For example `'2019_02_27/0002'`. For kitti-360 is of format `<sequence_id>`. For example `'2013_05_28_drive_0000'`.

`<parameter file>` defines the parameter file to be used (`params_cadcd.yaml` or `params_kitti360.yaml`).  

The sampled data is saved under `<dataset_path>/processed/<sequence>/data`.


### 3. Pre-process
Run the pre-processing script:
```
python3 pre_processing.py <sequence> <parameter file>
```
The outputs are saved under `<dataset_path>/processed/<sequence>/autolabels/pre_processing`.

<img src="https://github.com/user-attachments/assets/ebcf04b3-52dd-4965-9d07-72a9048e9fe7" alt="Image description" width="500"/>

### 4. Lidar autolabel
Run the lidar-autolabeling script:
```
python3 lidar_autolabeling.py <sequence> <parameter file>
```
The outputs are saved under `<dataset_path>/processed/<sequence>/autolabels/lidar`.
<img src="https://github.com/user-attachments/assets/3f3106c0-7a51-4e14-8c02-c688fae93666" alt="Image description" width="500"/>

### 5. Camera autolabel
Run the visual autolabeling script:
```
python3 camera_autolabeling.py <sequence> <parameter file>
```
The outputs are saved under `<dataset_path>/processed/<sequence>/autolabels/camera`.
<img src="https://github.com/user-attachments/assets/228397e2-43b4-4cae-8c9c-18cecc463585" alt="Image description" width="500"/>

### 6. Post-process
Run the post-processing script:
```
python3 post_processing.py <sequence> <parameter file>
```
The outputs are saved under `<dataset_path>/processed/<sequence>/autolabels/post_processing`.
<img src="https://github.com/user-attachments/assets/60d7008a-81b4-4e3c-80e9-9060d93319b0" alt="Image description" width="500"/>

## Citing

If you use our code or the related paper in your research, please consider citing.

```bibtex
```



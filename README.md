# lidar-camera-road-autolabeling

Source code for the paper: "Trajectory-based road auto labeling with lidar-camera fusion"

## Installation

```
git clone git@github.com:eerik98/lidar-camera-road-autolabeling.git
cd lidar-camera-road-autolabeling
conda env create -f conda.yaml
conda activate lidar-camera-road-autolabeling
```

## Usage
### 1. Download data
Download the open-source CADCD raw dataset.[`Homepage`](http://cadcd.uwaterloo.ca) [`Devkit`](https://github.com/mpitropov/cadc_devkit)

In `cadcd/raw/<day>/calib/extrinsics.yaml` `T_LIDAR_GPSIMU` change the last column of the second row from `0.0726...` to `-1.5`. Otherwise, the trajectory will be misaligned. We believe there is an error in the provided calibration matrix. 

### 2. Sample
Change the `dataset_path` in `params_cadcd.yaml` to the path of the downloaded dataset. Then run the sampling script: 
```
python3 sample_cadcd.py <day> <sequence>
```
`<day>` and `<sequence>` define the sequence to be processed. For example `'2019_02_27'` `'0002'`. The sampled data is saved under `processed/<day>/<seq>/data`.
### 3. Pre-process
Run the pre-processing script:
```
python3 pre_processing.py <day> <sequence>
```
The outputs are saved under `processed/<day>/<seq>/autolabels/pre_processing`.

<img src="https://github.com/user-attachments/assets/ebcf04b3-52dd-4965-9d07-72a9048e9fe7" alt="Image description" width="500"/>

### 4. Lidar autolabel
Run the lidar-autolabeling script:
```
python3 lidar_autolabeling.py <day> <sequence>
```
The outputs are saved under `processed/<day>/<seq>/autolabels/lidar`.
<img src="https://github.com/user-attachments/assets/3f3106c0-7a51-4e14-8c02-c688fae93666" alt="Image description" width="500"/>

### 5. Camera autolabel
Run the visual autolabeling script:
```
python3 camera_autolabeling.py <day> <sequence>
```
The outputs are saved under `processed/<day>/<seq>/autolabels/camera`.
<img src="https://github.com/user-attachments/assets/228397e2-43b4-4cae-8c9c-18cecc463585" alt="Image description" width="500"/>

### 6. Post-process
Run the post-processing script:
```
python3 post_processing.py <day> <sequence>
```
The outputs are saved under `processed/<day>/<seq>/autolabels/post_processing`.
<img src="https://github.com/user-attachments/assets/60d7008a-81b4-4e3c-80e9-9060d93319b0" alt="Image description" width="500"/>

## Citing

If you use our code or the related paper in your research, please consider citing.

```bibtex
```



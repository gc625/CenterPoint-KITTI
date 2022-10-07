## Package for SAIC data processing



#### 1. Data for downloading

1. **LiDAR data**  Location: /input/lidar/, Format: 10Hz, .pcd, seperate files

2.  **Radar data**  Location: /input/raw/, Format: .csv, needs code for extraction, one file for all
3.  **Pose data** Location: /output/online/sample/gnssimu-sample... Format: 100 Hz, needs conversion
4.  **GT data** Locations /output/obj_gt/GT.csv,  Format: needs code for extraction,  10Hz, same with LiDAR
5.  **Meta data** Location: /meta.xml. "start_posix_local" is the base timestamp for pose and radar data, "start_posix_utc" is the base timestamp for LiDAR and GT data.

Put all needed data under the root path.



#### 2. Extrinsic Parameters 

| Sensor        | x (m) | y (m) | z (m) | yaw (°) | pitch (°) | roll (°) |
| ------------- | ----- | ----- | ----- | ------- | --------- | -------- |
| LiDAR         | 0     | 0     | -0.3  | -2.5    | 0         | 0        |
| Radar (front) | 0.06  | -0.2  | 0.7   | -3.5    | 2         | 180      |

These parameters are adjusted by us and can roughly align LiDAR, radar point cloud and GT bounding boxes.



#### 3. Usage of package

1. read_raw.py: This file is used for extract a sequence of radar sweeps in the raw.csv file. 
2. pose_extract.py: This file is used for extract gnssimu data and convert them to ego poses.
3. extract_gt.py: This file is used for extract a sequence of GT from the GT.csv file. 
4. pcl_sync.py: This file is used to synchronize radar data with LiDAR data. The about 12Hz radar data is matched to nearest LiDAR sweep. We use pose data to compensation this temporal gap. As a result, we got a 10Hz radar data synchronized with LiDAR sweeps. Moreover, we compensate some delays of GT data and synchronize them to LiDAR data.
5. vis_pcl.py: This file is used to visualize the final Radar, Lidar and GT data. 

Please run the files w.r.t the order above. You can use /sync_radar, /sync_gt, and the original /lidar/../ folder for your task


# ORB_SLAM_live-map
Build live frequency occupy grid map from ORB_SLAM map points. Also has the option to generate new map points base on ORB_SLAM map points to make the map denser. 

This repository use a modified version of ORB_SLAM that output the map points and keyframes to disk: https://github.com/minhnhat93/ORB_SLAM2

For details how the extra map points are generated see the slide: https://github.com/minhnhat93/ORB_SLAM_live-map/tree/master/docs/slide_extra_map_points_for_orb_slam.odp

# Segementation + Illustration of ORB_SLAM map points (green) and extra map points (red):
![alt tag](https://github.com/minhnhat93/ORB_SLAM_live-map/tree/master/images/segmentation_with_mps_and_extra_points.png)

# Occupancy Grid Map building with ORB_SLAM map points:
![alt tag](https://github.com/minhnhat93/ORB_SLAM_live-map/tree/master/images/map_thresholded.png)

# Occupancy Grid Map building with ORB_SLAM map points and extra map points:
![alt tag](https://github.com/minhnhat93/ORB_SLAM_live-map/tree/master/images/map_thresholded_with_extra_points.png)

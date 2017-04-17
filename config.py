from os.path import join, expanduser

fx = 535.4
fy = 539.2
cx = 320.1
cy = 247.6
scalingFactor = 5000.0

ORB_SLAM2_DIR = expanduser('/home/nhat/git-clones/ORB_SLAM2')
KEYFRAME_DIR = join(ORB_SLAM2_DIR, 'KeyFrames')
MAPPOINT_DIR = join(ORB_SLAM2_DIR, 'MapPoints')
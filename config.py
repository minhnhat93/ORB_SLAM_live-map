from os.path import join, expanduser

# our camera
# fx = 591.641742
# fy = 585.372837
# cx = 349.367367
# cy = 242.382600
#
# k1 = -0.040173
# k2 = -0.267639
# p1 = 0.012474
# p2 = 0.004509
# k3 = 0.000000

# tum
fx = 535.4
fy = 539.2
cx = 320.1
cy = 247.6

k1 = 0.0
k2 = 0.0
p1 = 0.0
p2 = 0.0

scalingFactor = 5000.0

ORB_SLAM2_DIR = expanduser('/home/nhat/git-clones/ORB_SLAM2')
KEYFRAME_DIR = join(ORB_SLAM2_DIR, 'KeyFrames')
MAPPOINT_DIR = join(ORB_SLAM2_DIR, 'MapPoints')
import numpy as np
from sklearn.cluster import KMeans
from scipy.misc import imresize
from config import fx, fy, cx, cy, KEYFRAME_DIR
import math
import matplotlib.pyplot as plt
from scipy import ndimage
import time
import os
from os.path import join
from PIL import Image


def read_keyframes(path):
  return np.loadtxt(path, dtype=dict(
    names=('time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'),
    formats=('|S17', np.float, np.float, np.float, np.float, np.float, np.float, np.float)
  ), delimiter=' ', skiprows=0)

def read_mappoints(path):
  return np.loadtxt(path, dtype=dict(
    names=('time', 'id', 'x', 'y', 'z'),
    formats=('|S17', np.float, np.float, np.float, np.float)
  ), delimiter=' ', skiprows=0)

def read_until_good(path, read_method):
  while 1:
    try:
      out = read_method(path)
      if out[-1][0] == 'DONE' or out[-1][0] == 'END':
        if out[-1][0] == 'END':
          isFinished = True
        else:
          isFinished = False
        out = out[:-1]
        return out, isFinished
      time.sleep(0.1)
    except:
      pass

def get_graph_segment_for_frame(frameID, image_id=0, sigma=1.0, k=200, min=50):
  fn = join(KEYFRAME_DIR, '{:.6f}'.format(frameID) + '_' + str(image_id) + '.ppm')
  print(fn)
  return get_graph_segment(fn, sigma, k, min)

def get_graph_segment(fn, sigma=1.0, k=500, min=50):
  os.system('segment/segment {} {} {} {} {}'.format(sigma, k, min, fn, 'segment/out.ppm'))
  out = np.asarray(Image.open('segment/out.ppm'))
  return out

def get_num_cols_rows(origins, resolution):
  numCols = int(math.ceil((origins[2] - origins[0]) / resolution)) + 1
  numRows = int(math.ceil((origins[3] - origins[1]) / resolution)) + 1
  return numCols, numRows


def depth_cluster_1(depth_map, n_clusters=10, downsample=5, init='k-means++'):
  depth_downsampled = depth_map[::downsample, ::downsample]
  depth_flatten = depth_downsampled.reshape((depth_downsampled.size, 1))
  clt = KMeans(n_clusters=n_clusters, init=init)
  clt.fit(depth_flatten)
  return imresize(clt.labels_.reshape(depth_downsampled.shape), depth_map.shape)

def depth_cluster_2(depth_map, rgb_downsampled, orb_depth, downsample=5, depth_ratio=1.0, rgb_ratio=1.0, position_ratio=0.1, remove_noise=False):
  depth_downsampled = imresize(depth_map, (depth_map.shape[0] / downsample, depth_map.shape[1] / downsample), interp='bicubic')
  rgb_downsampled = imresize(rgb_downsampled, (rgb_downsampled.shape[0] / downsample, rgb_downsampled.shape[1] / downsample), interp='bicubic')
  rgb_r = rgb_downsampled[:, :, 0].reshape((rgb_downsampled.shape[0] * rgb_downsampled.shape[1],))
  rgb_g = rgb_downsampled[:, :, 1].reshape((rgb_downsampled.shape[0] * rgb_downsampled.shape[1],))
  rgb_b = rgb_downsampled[:, :, 2].reshape((rgb_downsampled.shape[0] * rgb_downsampled.shape[1],))
  depth_flatten = depth_downsampled.reshape((depth_downsampled.size,))
  x = np.arange(0, depth_downsampled.shape[1], 1).flatten()
  y = np.arange(0, depth_downsampled.shape[0], 1).flatten()
  xx, yy = np.meshgrid(x, y, sparse=False)
  xx = xx.reshape((xx.size))
  yy = yy.reshape((yy.size))
  fit_data = np.stack((depth_flatten * depth_ratio, xx * position_ratio, yy * position_ratio,
                       rgb_r * rgb_ratio, rgb_g * rgb_ratio, rgb_b * rgb_ratio), axis=-1)
  xx_init, yy_init = np.where(orb_depth > 0.0)
  xx_init /= downsample
  yy_init /= downsample
  depth_init = depth_downsampled[(xx_init, yy_init)]
  rgb_init = rgb_downsampled[(xx_init, yy_init)]
  xx_init = xx_init.reshape((xx_init.size,))
  yy_init = yy_init.reshape((yy_init.size,))
  fit_init = np.stack((depth_init * depth_ratio, xx_init * position_ratio, yy_init * position_ratio,
                       rgb_init[:, 0] * rgb_ratio, rgb_init[:, 1] * rgb_ratio, rgb_init[:, 2] * rgb_ratio), axis=-1)
  clt = KMeans(n_clusters=fit_init.shape[0], init=fit_init)
  clt.fit(fit_data)
  _result = clt.labels_.reshape(depth_downsampled.shape)
  if remove_noise:
    structure = np.ones((3, 3))
    labels = np.unique(clt.labels_)
    for label in labels:
      mask = (_result == label)
      eroded_mask = ndimage.binary_erosion(mask, structure)
      border = (mask != eroded_mask)
      _result[border] = 0
  return imresize(_result, depth_map.shape)

def create_depthmap_from_pointcloud(map_points, keyframe, im_size=None):
  if im_size is None:
    im_size = (480, 640)
  _depth = np.zeros(im_size)
  for point in map_points:
    dx, dy, dz =\
      [point[0] - keyframe[0],
       point[1] - keyframe[1],
       point[2] - keyframe[2]]
    u = dx * fx / dz + cx
    v = dy * fy / dz + cy
    if v > -1 and v < im_size[0] and u > -1 and u < im_size[1]:
      _depth[v, u] = dz
    #else:
    #  print(v, u)
  return _depth

def merge_depth_with_segmentation(segmentation, gt_depth, mean='average'):
  labels = np.unique(segmentation)
  _depth = np.zeros(segmentation.shape)
  for label in labels:
    mask = (segmentation == label)
    if mean == 'average':
      count = (gt_depth[mask] > 0).astype(float).sum()
      sum = gt_depth[mask].sum()
      average = sum / count
      #if count > 0:
      #  plt.imshow(mask)
      #  plt.show()
    elif mean == 'harmonic':
      intersection_set = 1
      average = 1
    else:
      raise Exception
    _depth[mask] = average
    _depth[np.isnan(_depth)] = 0
  return _depth


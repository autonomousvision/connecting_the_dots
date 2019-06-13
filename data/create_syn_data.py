import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
from pathlib import Path
import multiprocessing
import time
import json
import cv2
import os
import collections
import sys
sys.path.append('../')
import renderer
import co
from commons import get_patterns,get_rotation_matrix
from lcn import lcn

def get_objs(shapenet_dir, obj_classes, num_perclass=100):

  shapenet = {'chair':      '03001627',
              'airplane':   '02691156',
              'car':        '02958343',
              'watercraft': '04530566'}

  obj_paths = []
  for cls in obj_classes:
      if cls not in shapenet.keys():
          raise Exception('unknown class name')
      ids = shapenet[cls]
      obj_path = sorted(Path(f'{shapenet_dir}/{ids}').glob('**/models/*.obj'))
      obj_paths += obj_path[:num_perclass]
  print(f'found {len(obj_paths)} object paths')

  objs = []
  for obj_path in obj_paths:
    print(f'load {obj_path}')
    v, f, _, n = co.io3d.read_obj(obj_path)
    diffs = v.max(axis=0) - v.min(axis=0)
    v /= (0.5 * diffs.max())
    v -= (v.min(axis=0) + 1)
    f = f.astype(np.int32)
    objs.append((v,f,n))
  print(f'loaded {len(objs)} objects')

  return objs


def get_mesh(rng, min_z=0):
  # set up background board
  verts, faces, normals, colors = [], [], [], []
  v, f, n = co.geometry.xyplane(z=0, interleaved=True)
  v[:,2] += -v[:,2].min() + rng.uniform(2,7)
  v[:,:2] *= 5e2
  v[:,2] = np.mean(v[:,2]) + (v[:,2] - np.mean(v[:,2])) * 5e2
  c = np.empty_like(v)
  c[:] = rng.uniform(0,1, size=(3,)).astype(np.float32)
  verts.append(v)
  faces.append(f)
  normals.append(n)
  colors.append(c)

  # randomly sample 4 foreground objects for each scene
  for shape_idx in range(4):
    v, f, n = objs[rng.randint(0,len(objs))]
    v, f, n = v.copy(), f.copy(), n.copy()

    s = rng.uniform(0.25, 1)
    v *= s
    R = co.geometry.rotm_from_quat(co.geometry.quat_random(rng=rng))
    v = v @ R.T
    n = n @ R.T
    v[:,2] += -v[:,2].min() + min_z + rng.uniform(0.5, 3)
    v[:,:2] += rng.uniform(-1, 1, size=(1,2))

    c = np.empty_like(v)
    c[:] = rng.uniform(0,1, size=(3,)).astype(np.float32)

    verts.append(v.astype(np.float32))
    faces.append(f)
    normals.append(n)
    colors.append(c)

  verts, faces = co.geometry.stack_mesh(verts, faces)
  normals = np.vstack(normals).astype(np.float32)
  colors = np.vstack(colors).astype(np.float32)
  return verts, faces, colors, normals


def create_data(out_root, idx, n_samples, imsize, patterns, K, baseline, blend_im, noise, track_length=4):

  tic = time.time()
  rng = np.random.RandomState()

  rng.seed(idx)

  verts, faces, colors, normals = get_mesh(rng)
  data = renderer.PyRenderInput(verts=verts.copy(), colors=colors.copy(), normals=normals.copy(), faces=faces.copy())
  print(f'loading mesh for sample {idx+1}/{n_samples} took {time.time()-tic}[s]')


  # let the camera point to the center
  center = np.array([0,0,3], dtype=np.float32)

  basevec =  np.array([-baseline,0,0], dtype=np.float32)
  unit = np.array([0,0,1],dtype=np.float32)

  cam_x_ = rng.uniform(-0.2,0.2)
  cam_y_ = rng.uniform(-0.2,0.2)
  cam_z_ = rng.uniform(-0.2,0.2)

  ret = collections.defaultdict(list)
  blend_im_rnd = np.clip(blend_im + rng.uniform(-0.1,0.1), 0,1)

  # capture the same static scene from different view points as a track
  for ind in range(track_length):

    cam_x = cam_x_ + rng.uniform(-0.1,0.1)
    cam_y = cam_y_ + rng.uniform(-0.1,0.1)
    cam_z = cam_z_ + rng.uniform(-0.1,0.1)
    
    tcam = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    if np.linalg.norm(tcam[0:2])<1e-9:
      Rcam = np.eye(3, dtype=np.float32) 
    else:
      Rcam = get_rotation_matrix(center, center-tcam)

    tproj = tcam + basevec 
    Rproj = Rcam 

    ret['R'].append(Rcam)
    ret['t'].append(tcam)

    cams = []
    projs = []

    # render the scene at multiple scales
    scales = [1, 0.5, 0.25, 0.125]

    for scale in scales:
      fx = K[0,0] * scale
      fy = K[1,1] * scale
      px = K[0,2] * scale
      py = K[1,2] * scale
      im_height = imsize[0] * scale
      im_width = imsize[1] * scale
      cams.append( renderer.PyCamera(fx,fy,px,py, Rcam, tcam, im_width, im_height) )
      projs.append( renderer.PyCamera(fx,fy,px,py, Rproj, tproj, im_width, im_height) )


    for s, cam, proj, pattern in zip(itertools.count(), cams, projs, patterns):
      fl = K[0,0] / (2**s)

      shader = renderer.PyShader(0.5,1.5,0.0,10)
      pyrenderer = renderer.PyRenderer(cam, shader, engine='gpu')
      pyrenderer.mesh_proj(data, proj, pattern, d_alpha=0, d_beta=0.35)

      # get the reflected laser pattern $R$
      im = pyrenderer.color().copy()
      depth = pyrenderer.depth().copy()
      disp = baseline * fl / depth
      mask = depth > 0
      im = np.mean(im, axis=2)

      # get the ambient image $A$
      ambient = pyrenderer.normal().copy()
      ambient = np.mean(ambient, axis=2)

      # get the noise free IR image $J$ 
      im = blend_im_rnd * im + (1 - blend_im_rnd) * ambient
      ret[f'ambient{s}'].append( ambient[None].astype(np.float32) )

      # get the gradient magnitude of the ambient image $|\nabla A|$
      ambient = ambient.astype(np.float32)
      sobelx = cv2.Sobel(ambient,cv2.CV_32F,1,0,ksize=5)
      sobely = cv2.Sobel(ambient,cv2.CV_32F,0,1,ksize=5)
      grad = np.sqrt(sobelx**2 + sobely**2)
      grad = np.maximum(grad-0.8,0.0) # parameter

      # get the local contract normalized grad LCN($|\nabla A|$)
      grad_lcn, grad_std = lcn.normalize(grad,5,0.1)
      grad_lcn = np.clip(grad_lcn,0.0,1.0) # parameter
      ret[f'grad{s}'].append( grad_lcn[None].astype(np.float32))

      ret[f'im{s}'].append( im[None].astype(np.float32))
      ret[f'mask{s}'].append(mask[None].astype(np.float32))
      ret[f'disp{s}'].append(disp[None].astype(np.float32))

  for key in ret.keys():
    ret[key] = np.stack(ret[key], axis=0)

  # save to files
  out_dir = out_root / f'{idx:08d}'
  out_dir.mkdir(exist_ok=True, parents=True)
  for k,val in ret.items():
    for tidx in range(track_length):
      v = val[tidx]
      out_path = out_dir / f'{k}_{tidx}.npy'
      np.save(out_path, v)
  np.save( str(out_dir /'blend_im.npy'), blend_im_rnd)

  print(f'create sample {idx+1}/{n_samples} took {time.time()-tic}[s]')



if __name__=='__main__':

  np.random.seed(42)
  
  # output directory
  with open('../config.json') as fp:
   config = json.load(fp)
   data_root = Path(config['DATA_ROOT'])
   shapenet_root = config['SHAPENET_ROOT']
  
  data_type = 'syn'
  out_root = data_root / f'{data_type}'
  out_root.mkdir(parents=True, exist_ok=True)

  # load shapenet models 
  obj_classes = ['chair']
  objs = get_objs(shapenet_root, obj_classes)
  
  # camera parameters
  imsize = (480, 640)
  imsizes = [(imsize[0]//(2**s), imsize[1]//(2**s)) for s in range(4)]
  K = np.array([[567.6, 0, 324.7], [0, 570.2, 250.1], [0 ,0, 1]], dtype=np.float32)
  focal_lengths = [K[0,0]/(2**s) for s in range(4)]
  baseline=0.075
  blend_im = 0.6
  noise = 0
  
  # capture the same static scene from different view points as a track
  track_length = 4
  
  # load pattern image
  pattern_path = './kinect_pattern.png'
  pattern_crop = True
  patterns = get_patterns(pattern_path, imsizes, pattern_crop)
  
  # write settings to file
  settings = {
   'imsizes': imsizes,
   'patterns': patterns,
   'focal_lengths': focal_lengths,
   'baseline': baseline,
   'K': K,
  }
  out_path = out_root / f'settings.pkl'
  print(f'write settings to {out_path}')
  with open(str(out_path), 'wb') as f:
   pickle.dump(settings, f, pickle.HIGHEST_PROTOCOL)
  
  # start the job
  n_samples = 2**10 + 2**13
  for idx in range(n_samples):
    args = (out_root, idx, n_samples, imsize, patterns, K, baseline, blend_im, noise, track_length)
    create_data(*args)

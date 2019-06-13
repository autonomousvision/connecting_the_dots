import numpy as np
import scipy.spatial
import scipy.linalg

def nullspace(A, atol=1e-13, rtol=0):
  u, s, vh = np.linalg.svd(A)
  tol = max(atol, rtol * s[0])
  nnz = (s >= tol).sum()
  ns = vh[nnz:].conj().T
  return ns

def nearest_orthogonal_matrix(R):
  U,S,Vt = np.linalg.svd(R)
  return U @ np.eye(3,dtype=R.dtype) @ Vt

def power_iters(A, n_iters=10):
  b = np.random.uniform(-1,1, size=(A.shape[0], A.shape[1], 1))
  for iter in range(n_iters):
    b = A @ b
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
  return b

def rayleigh_quotient(A, b):
  return (b.transpose(0,2,1) @ A @ b) / (b.transpose(0,2,1) @ b)


def cross_prod_mat(x):
  x = x.reshape(-1,3)
  X = np.empty((x.shape[0],3,3), dtype=x.dtype)
  X[:,0,0] = 0
  X[:,0,1] = -x[:,2]
  X[:,0,2] = x[:,1]
  X[:,1,0] = x[:,2]
  X[:,1,1] = 0
  X[:,1,2] = -x[:,0]
  X[:,2,0] = -x[:,1]
  X[:,2,1] = x[:,0]
  X[:,2,2] = 0
  return X.squeeze()

def hat_operator(x):
  return cross_prod_mat(x)

def vee_operator(X):
  X = X.reshape(-1,3,3)
  x = np.empty((X.shape[0], 3), dtype=X.dtype)
  x[:,0] = X[:,2,1]
  x[:,1] = X[:,0,2]
  x[:,2] = X[:,1,0]
  return x.squeeze()


def rot_x(x, dtype=np.float32):
  x = np.array(x, copy=False)
  x = x.reshape(-1,1)
  R = np.zeros((x.shape[0],3,3), dtype=dtype)
  R[:,0,0] = 1
  R[:,1,1] = np.cos(x).ravel()
  R[:,1,2] = -np.sin(x).ravel()
  R[:,2,1] = np.sin(x).ravel()
  R[:,2,2] = np.cos(x).ravel()
  return R.squeeze()

def rot_y(y, dtype=np.float32):
  y = np.array(y, copy=False)
  y = y.reshape(-1,1)
  R = np.zeros((y.shape[0],3,3), dtype=dtype)
  R[:,0,0] = np.cos(y).ravel()
  R[:,0,2] = np.sin(y).ravel()
  R[:,1,1] = 1
  R[:,2,0] = -np.sin(y).ravel()
  R[:,2,2] = np.cos(y).ravel()
  return R.squeeze()

def rot_z(z, dtype=np.float32):
  z = np.array(z, copy=False)
  z = z.reshape(-1,1)
  R = np.zeros((z.shape[0],3,3), dtype=dtype)
  R[:,0,0] = np.cos(z).ravel()
  R[:,0,1] = -np.sin(z).ravel()
  R[:,1,0] = np.sin(z).ravel()
  R[:,1,1] = np.cos(z).ravel()
  R[:,2,2] = 1
  return R.squeeze()

def xyz_from_rotm(R):
  R = R.reshape(-1,3,3)
  xyz = np.empty((R.shape[0],3), dtype=R.dtype)
  for bidx in range(R.shape[0]):
    if R[bidx,0,2] < 1:
      if R[bidx,0,2] > -1:
        xyz[bidx,1] = np.arcsin(R[bidx,0,2])
        xyz[bidx,0] = np.arctan2(-R[bidx,1,2], R[bidx,2,2])
        xyz[bidx,2] = np.arctan2(-R[bidx,0,1], R[bidx,0,0])
      else:
        xyz[bidx,1] = -np.pi/2
        xyz[bidx,0] = -np.arctan2(R[bidx,1,0],R[bidx,1,1])
        xyz[bidx,2] = 0
    else:
      xyz[bidx,1] = np.pi/2
      xyz[bidx,0] = np.arctan2(R[bidx,1,0], R[bidx,1,1])
      xyz[bidx,2] = 0
  return xyz.squeeze()

def zyx_from_rotm(R):
  R = R.reshape(-1,3,3)
  zyx = np.empty((R.shape[0],3), dtype=R.dtype)
  for bidx in range(R.shape[0]):
    if R[bidx,2,0] < 1:
      if R[bidx,2,0] > -1:
        zyx[bidx,1] = np.arcsin(-R[bidx,2,0])
        zyx[bidx,0] = np.arctan2(R[bidx,1,0], R[bidx,0,0])
        zyx[bidx,2] = np.arctan2(R[bidx,2,1], R[bidx,2,2])
      else:
        zyx[bidx,1] = np.pi / 2
        zyx[bidx,0] = -np.arctan2(-R[bidx,1,2], R[bidx,1,1])
        zyx[bidx,2] = 0
    else:
      zyx[bidx,1] = -np.pi / 2
      zyx[bidx,0] = np.arctan2(-R[bidx,1,2], R[bidx,1,1])
      zyx[bidx,2] = 0
  return zyx.squeeze()

def rotm_from_xyz(xyz):
  xyz = np.array(xyz, copy=False).reshape(-1,3)
  return (rot_x(xyz[:,0]) @ rot_y(xyz[:,1]) @ rot_z(xyz[:,2])).squeeze()

def rotm_from_zyx(zyx):
  zyx = np.array(zyx, copy=False).reshape(-1,3)
  return (rot_z(zyx[:,0]) @ rot_y(zyx[:,1]) @ rot_x(zyx[:,2])).squeeze()

def rotm_from_quat(q):
  q = q.reshape(-1,4)
  w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
  R = np.array([
    [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
    [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
    [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
  ], dtype=q.dtype)
  R = R.transpose((2,0,1))
  return R.squeeze()

def rotm_from_axisangle(a):
  # exponential
  a = a.reshape(-1,3)
  phi = np.linalg.norm(a, axis=1).reshape(-1,1,1)
  iphi = np.zeros_like(phi)
  np.divide(1, phi, out=iphi, where=phi != 0)
  A = cross_prod_mat(a) * iphi
  R = np.eye(3, dtype=a.dtype) + np.sin(phi) * A + (1 - np.cos(phi)) * A @ A
  return R.squeeze()

def rotm_from_lookat(dir, up=None):
  dir = dir.reshape(-1,3)
  if up is None:
    up = np.zeros_like(dir)
    up[:,1] = 1
  dir /= np.linalg.norm(dir, axis=1, keepdims=True)
  up /= np.linalg.norm(up, axis=1, keepdims=True)
  x = dir[:,None,:] @ cross_prod_mat(up).transpose(0,2,1)
  y = x @ cross_prod_mat(dir).transpose(0,2,1)
  x = x.squeeze()
  y = y.squeeze()
  x /= np.linalg.norm(x, axis=1, keepdims=True)
  y /= np.linalg.norm(y, axis=1, keepdims=True)
  R = np.empty((dir.shape[0],3,3), dtype=dir.dtype)
  R[:,0,0] = x[:,0]
  R[:,0,1] = y[:,0]
  R[:,0,2] = dir[:,0]
  R[:,1,0] = x[:,1]
  R[:,1,1] = y[:,1]
  R[:,1,2] = dir[:,1]
  R[:,2,0] = x[:,2]
  R[:,2,1] = y[:,2]
  R[:,2,2] = dir[:,2]
  return R.transpose(0,2,1).squeeze()

def rotm_distance_identity(R0, R1):
  # https://link.springer.com/article/10.1007%2Fs10851-009-0161-2
  # in [0, 2*sqrt(2)]
  R0 = R0.reshape(-1,3,3)
  R1 = R1.reshape(-1,3,3)
  dists = np.linalg.norm(np.eye(3,dtype=R0.dtype) - R0 @ R1.transpose(0,2,1), axis=(1,2))
  return dists.squeeze()

def rotm_distance_geodesic(R0, R1):
  # https://link.springer.com/article/10.1007%2Fs10851-009-0161-2
  # in [0, pi)
  R0 = R0.reshape(-1,3,3)
  R1 = R1.reshape(-1,3,3)
  RtR = R0 @ R1.transpose(0,2,1)
  aa = axisangle_from_rotm(RtR)
  S = cross_prod_mat(aa).reshape(-1,3,3)
  dists = np.linalg.norm(S, axis=(1,2))
  return dists.squeeze()



def axisangle_from_rotm(R):
  # logarithm of rotation matrix
  # R = R.reshape(-1,3,3)
  # tr = np.trace(R, axis1=1, axis2=2)
  # phi = np.arccos(np.clip((tr - 1) / 2, -1, 1))
  # scale = np.zeros_like(phi)
  # div = 2 * np.sin(phi)
  # np.divide(phi, div, out=scale, where=np.abs(div) > 1e-6)
  # A = (R - R.transpose(0,2,1)) * scale.reshape(-1,1,1)
  # aa = np.stack((A[:,2,1], A[:,0,2], A[:,1,0]), axis=1)
  # return aa.squeeze()
  R = R.reshape(-1,3,3)
  omega = np.empty((R.shape[0], 3), dtype=R.dtype)
  omega[:,0] = R[:,2,1] - R[:,1,2]
  omega[:,1] = R[:,0,2] - R[:,2,0]
  omega[:,2] = R[:,1,0] - R[:,0,1]
  r = np.linalg.norm(omega, axis=1).reshape(-1,1)
  t = np.trace(R, axis1=1, axis2=2).reshape(-1,1)
  omega = np.arctan2(r, t-1) * omega
  aa = np.zeros_like(omega)
  np.divide(omega, r, out=aa, where=r != 0)
  return aa.squeeze()

def axisangle_from_quat(q):
  q = q.reshape(-1,4)
  phi = 2 * np.arccos(q[:,0])
  denom = np.zeros_like(q[:,0])
  np.divide(1, np.sqrt(1 - q[:,0]**2), out=denom, where=q[:,0] != 1)
  axis = q[:,1:] * denom.reshape(-1,1)
  denom = np.linalg.norm(axis, axis=1).reshape(-1,1)
  a = np.zeros_like(axis)
  np.divide(phi.reshape(-1,1) * axis, denom, out=a, where=denom != 0)
  aa = a.astype(q.dtype)
  return aa.squeeze()

def axisangle_apply(aa, x):
  # working only with single aa and single x at the moment
  xshape = x.shape
  aa = aa.reshape(3,)
  x = x.reshape(3,)
  phi = np.linalg.norm(aa)
  e = np.zeros_like(aa)
  np.divide(aa, phi, out=e, where=phi != 0)
  xr = np.cos(phi) * x + np.sin(phi) * np.cross(e, x) + (1 - np.cos(phi)) * (e.T @ x) * e
  return xr.reshape(xshape)


def exp_so3(R):
  w = axisangle_from_rotm(R)
  return w

def log_so3(w):
  R = rotm_from_axisangle(w)
  return R

def exp_se3(R, t):
  R = R.reshape(-1,3,3)
  t = t.reshape(-1,3)

  w = exp_so3(R).reshape(-1,3)

  phi = np.linalg.norm(w, axis=1).reshape(-1,1,1)
  A = cross_prod_mat(w)
  Vi = np.eye(3, dtype=R.dtype) - A/2 + (1 - (phi * np.sin(phi) / (2 * (1 - np.cos(phi))))) / phi**2 * A @ A
  u = t.reshape(-1,1,3) @ Vi.transpose(0,2,1)

  # v = (u, w)
  v = np.empty((R.shape[0],6), dtype=R.dtype)
  v[:,:3] = u.squeeze()
  v[:,3:] = w

  return v.squeeze()

def log_se3(v):
  # v = (u, w)
  v = v.reshape(-1,6)
  u = v[:,:3]
  w = v[:,3:]

  R = log_so3(w)

  phi = np.linalg.norm(w, axis=1).reshape(-1,1,1)
  A = cross_prod_mat(w)
  V = np.eye(3, dtype=v.dtype) + (1 - np.cos(phi)) / phi**2 * A + (phi - np.sin(phi)) / phi**3 * A @ A
  t = u.reshape(-1,1,3) @ V.transpose(0,2,1)

  return R.squeeze(), t.squeeze()


def quat_from_rotm(R):
  R = R.reshape(-1,3,3)
  q = np.empty((R.shape[0], 4,), dtype=R.dtype)
  q[:,0] = np.sqrt( np.maximum(0, 1 + R[:,0,0] + R[:,1,1] + R[:,2,2]) )
  q[:,1] = np.sqrt( np.maximum(0, 1 + R[:,0,0] - R[:,1,1] - R[:,2,2]) )
  q[:,2] = np.sqrt( np.maximum(0, 1 - R[:,0,0] + R[:,1,1] - R[:,2,2]) )
  q[:,3] = np.sqrt( np.maximum(0, 1 - R[:,0,0] - R[:,1,1] + R[:,2,2]) )
  q[:,1] *= np.sign(q[:,1] * (R[:,2,1] - R[:,1,2]))
  q[:,2] *= np.sign(q[:,2] * (R[:,0,2] - R[:,2,0]))
  q[:,3] *= np.sign(q[:,3] * (R[:,1,0] - R[:,0,1]))
  q /= np.linalg.norm(q,axis=1,keepdims=True)
  return q.squeeze()

def quat_from_axisangle(a):
  a = a.reshape(-1, 3)
  phi = np.linalg.norm(a, axis=1)
  iphi = np.zeros_like(phi)
  np.divide(1, phi, out=iphi, where=phi != 0)
  a = a * iphi.reshape(-1,1)
  theta = phi / 2.0
  r = np.cos(theta)
  stheta = np.sin(theta)
  q = np.stack((r, stheta*a[:,0], stheta*a[:,1], stheta*a[:,2]), axis=1)
  q /= np.linalg.norm(q, axis=1).reshape(-1,1)
  return q.squeeze()

def quat_identity(n=1, dtype=np.float32):
  q = np.zeros((n,4), dtype=dtype)
  q[:,0] = 1
  return q.squeeze()

def quat_conjugate(q):
  shape = q.shape
  q = q.reshape(-1,4).copy()
  q[:,1:] *= -1
  return q.reshape(shape)

def quat_product(q1, q2):
  # q1 . q2 is equivalent to R(q1) @ R(q2)
  shape = q1.shape
  q1, q2 = q1.reshape(-1,4), q2.reshape(-1, 4)
  q = np.empty((max(q1.shape[0], q2.shape[0]), 4), dtype=q1.dtype)
  a1,b1,c1,d1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
  a2,b2,c2,d2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
  q[:,0] = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
  q[:,1] = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
  q[:,2] = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
  q[:,3] = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
  return q.squeeze()

def quat_apply(q, x):
  xshape = x.shape
  x = x.reshape(-1, 3)
  qshape = q.shape
  q = q.reshape(-1, 4)

  p = np.empty((x.shape[0], 4), dtype=x.dtype)
  p[:,0] = 0
  p[:,1:] = x

  r = quat_product(quat_product(q, p), quat_conjugate(q))
  if r.ndim == 1:
    return r[1:].reshape(xshape)
  else:
    return r[:,1:].reshape(xshape)


def quat_random(rng=None, n=1):
  # http://planning.cs.uiuc.edu/node198.html
  if rng is not None:
    u = rng.uniform(0, 1, size=(3,n))
  else:
    u = np.random.uniform(0, 1, size=(3,n))
  q = np.array((
    np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
    np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
    np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
    np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
  )).T
  q /= np.linalg.norm(q,axis=1,keepdims=True)
  return q.squeeze()

def quat_distance_angle(q0, q1):
  # https://math.stackexchange.com/questions/90081/quaternion-distance
  # https://link.springer.com/article/10.1007%2Fs10851-009-0161-2
  q0 = q0.reshape(-1,4)
  q1 = q1.reshape(-1,4)
  dists = np.arccos(np.clip(2 * np.sum(q0 * q1, axis=1)**2 - 1, -1, 1))
  return dists

def quat_distance_normdiff(q0, q1):
  # https://link.springer.com/article/10.1007%2Fs10851-009-0161-2
  # \phi_4
  # [0, 1]
  q0 = q0.reshape(-1,4)
  q1 = q1.reshape(-1,4)
  return 1 - np.sum(q0 * q1, axis=1)**2

def quat_distance_mineucl(q0, q1):
  # https://link.springer.com/article/10.1007%2Fs10851-009-0161-2
  # http://users.cecs.anu.edu.au/~trumpf/pubs/Hartley_Trumpf_Dai_Li.pdf
  q0 = q0.reshape(-1,4)
  q1 = q1.reshape(-1,4)
  diff0 = ((q0 - q1)**2).sum(axis=1)
  diff1 = ((q0 + q1)**2).sum(axis=1)
  return np.minimum(diff0, diff1)

def quat_slerp_space(q0, q1, num=100, endpoint=True):
  q0 = q0.ravel()
  q1 = q1.ravel()
  dot = q0.dot(q1)
  if dot < 0:
    q1 *= -1
    dot *= -1
  t = np.linspace(0, 1, num=num, endpoint=endpoint, dtype=q0.dtype)
  t = t.reshape((-1,1))
  if dot > 0.9995:
    ret = q0 + t * (q1 - q0)
    return ret
  dot = np.clip(dot, -1, 1)
  theta0 = np.arccos(dot)
  theta = theta0 * t
  s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta0)
  s1 = np.sin(theta) / np.sin(theta0)
  return (s0 * q0) + (s1 * q1)

def cart_to_spherical(x):
  shape = x.shape
  x = x.reshape(-1,3)
  y = np.empty_like(x)
  y[:,0] = np.linalg.norm(x, axis=1)  # r
  y[:,1] = np.arccos(x[:,2] / y[:,0]) # theta
  y[:,2] = np.arctan2(x[:,1], x[:,0]) # phi
  return y.reshape(shape)

def spherical_to_cart(x):
  shape = x.shape
  x = x.reshape(-1,3)
  y = np.empty_like(x)
  y[:,0] = x[:,0] * np.sin(x[:,1]) * np.cos(x[:,2])
  y[:,1] = x[:,0] * np.sin(x[:,1]) * np.sin(x[:,2])
  y[:,2] = x[:,0] * np.cos(x[:,1])
  return y.reshape(shape)

def spherical_random(r=1, n=1):
  # http://mathworld.wolfram.com/SpherePointPicking.html
  # https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
  x = np.empty((n,3))
  x[:,0] = r
  x[:,1] = 2 * np.pi * np.random.uniform(0,1, size=(n,))
  x[:,2] = np.arccos(2 * np.random.uniform(0,1, size=(n,)) - 1)
  return x.squeeze()

def color_pcl(pcl, K, im, color_axis=0, as_int=True, invalid_color=[0,0,0]):
  uvd = K @ pcl.T
  uvd /= uvd[2]
  uvd = np.round(uvd).astype(np.int32)
  mask = np.logical_and(uvd[0] >= 0, uvd[1] >= 0)
  color = np.empty((pcl.shape[0], 3), dtype=im.dtype)
  if color_axis == 0:
    mask = np.logical_and(mask, uvd[0] < im.shape[2])
    mask = np.logical_and(mask, uvd[1] < im.shape[1])
    uvd = uvd[:,mask]
    color[mask,:] = im[:,uvd[1],uvd[0]].T
  elif color_axis == 2:
    mask = np.logical_and(mask, uvd[0] < im.shape[1])
    mask = np.logical_and(mask, uvd[1] < im.shape[0])
    uvd = uvd[:,mask]
    color[mask,:] = im[uvd[1],uvd[0], :]
  else:
    raise Exception('invalid color_axis')
  color[np.logical_not(mask),:3] = invalid_color
  if as_int:
    color = (255.0 * color).astype(np.int32)
  return color

def center_pcl(pcl, robust=False, copy=False, axis=1):
  if copy:
    pcl = pcl.copy()
  if robust:
    mu = np.median(pcl, axis=axis, keepdims=True)
  else:
    mu = np.mean(pcl, axis=axis, keepdims=True)
  return pcl - mu

def to_homogeneous(x):
  # return np.hstack((x, np.ones((x.shape[0],1),dtype=x.dtype)))
  return np.concatenate((x, np.ones((*x.shape[:-1],1),dtype=x.dtype)), axis=-1)

def from_homogeneous(x):
  return x[:,:-1] / x[:,-1]

def project_uvn(uv, Ki=None):
  if uv.shape[1] == 2:
    uvn = to_homogeneous(uv)
  else:
    uvn = uv
  if uvn.shape[1] != 3:
    raise Exception('uv should have shape Nx2 or Nx3')
  if Ki is None:
    return uvn
  else:
    return uvn @ Ki.T

def project_uvd(uv, depth, K=np.eye(3), R=np.eye(3), t=np.zeros((3,1)), ignore_negative_depth=True, return_uvn=False):
  Ki = np.linalg.inv(K)

  if ignore_negative_depth:
    mask = depth >= 0
    uv = uv[mask,:]
    d = depth[mask]
  else:
    d = depth.ravel()

  uv1 = to_homogeneous(uv)

  uvn1 = uv1 @ Ki.T
  xyz = d.reshape(-1,1) * uvn1
  xyz = (xyz - t.reshape((1,3))) @ R

  if return_uvn:
    return xyz, uvn1
  else:
    return xyz

def project_depth(depth, K, R=np.eye(3,3), t=np.zeros((3,1)), ignore_negative_depth=True, return_uvn=False):
  u, v = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
  uv = np.hstack((u.reshape(-1,1), v.reshape(-1,1)))
  return project_uvd(uv, depth.ravel(), K, R, t, ignore_negative_depth, return_uvn)


def project_xyz(xyz, K=np.eye(3), R=np.eye(3,3), t=np.zeros((3,1))):
  uvd = K @ (R @ xyz.T + t.reshape((3,1)))
  uvd[:2] /= uvd[2]
  return uvd[:2].T, uvd[2]


def relative_motion(R0, t0, R1, t1, Rt_from_global=True):
  t0 = t0.reshape((3,1))
  t1 = t1.reshape((3,1))
  if Rt_from_global:
    Rr = R1 @ R0.T
    tr = t1 - Rr @ t0
  else:
    Rr = R1.T @ R0
    tr = R1.T @ (t0 - t1)
  return Rr, tr.ravel()


def translation_to_cameracenter(R, t):
  t = t.reshape(-1,3,1)
  R = R.reshape(-1,3,3)
  C = -R.transpose(0,2,1) @ t
  return C.squeeze()

def cameracenter_to_translation(R, C):
  C = C.reshape(-1,3,1)
  R = R.reshape(-1,3,3)
  t = -R @ C
  return t.squeeze()

def decompose_projection_matrix(P, return_t=True):
  if P.shape[0] != 3 or P.shape[1] != 4:
    raise Exception('P has to be 3x4')
  M = P[:, :3]
  C = -np.linalg.inv(M) @ P[:, 3:]

  R,K = np.linalg.qr(np.flipud(M).T)
  K = np.flipud(K.T)
  K = np.fliplr(K)
  R = np.flipud(R.T)

  T = np.diag(np.sign(np.diag(K)))
  K = K @ T
  R = T @ R

  if np.linalg.det(R) < 0:
    R *= -1

  K /= K[2,2]
  if return_t:
    return K, R, cameracenter_to_translation(R, C)
  else:
    return K, R, C


def compose_projection_matrix(K=np.eye(3), R=np.eye(3,3), t=np.zeros((3,1))):
  return K @ np.hstack((R, t.reshape((3,1))))



def point_plane_distance(pts, plane):
  pts = pts.reshape(-1,3)
  return np.abs(np.sum(plane[:3] * pts, axis=1) + plane[3]) / np.linalg.norm(plane[:3])

def fit_plane(pts):
  pts = pts.reshape(-1,3)
  center = np.mean(pts, axis=0)
  A = pts - center
  u, s, vh = np.linalg.svd(A, full_matrices=False)
  # if pts.shape[0] > 100:
  #   import ipdb; ipdb.set_trace()
  plane = np.array([*vh[2], -vh[2].dot(center)])
  return plane

def tetrahedron(dtype=np.float32):
  verts = np.array([
    (np.sqrt(8/9), 0, -1/3), (-np.sqrt(2/9), np.sqrt(2/3), -1/3),
    (-np.sqrt(2/9), -np.sqrt(2/3), -1/3), (0, 0, 1)], dtype=dtype)
  faces = np.array([(0,1,2), (0,2,3), (0,1,3), (1,2,3)], dtype=np.int32)
  normals = -np.mean(verts, axis=0) + verts
  normals /= np.linalg.norm(normals, axis=1).reshape(-1,1)
  return verts, faces, normals

def cube(dtype=np.float32):
  verts = np.array([
    [-0.5,-0.5,-0.5], [-0.5,0.5,-0.5], [0.5,0.5,-0.5], [0.5,-0.5,-0.5],
    [-0.5,-0.5,0.5], [-0.5,0.5,0.5], [0.5,0.5,0.5], [0.5,-0.5,0.5]], dtype=dtype)
  faces = np.array([
    (0,1,2), (0,2,3), (4,5,6), (4,6,7),
    (0,4,7), (0,7,3), (1,5,6), (1,6,2),
    (3,2,6), (3,6,7), (0,1,5), (0,5,4)], dtype=np.int32)
  normals = -np.mean(verts, axis=0) + verts
  normals /= np.linalg.norm(normals, axis=1).reshape(-1,1)
  return verts, faces, normals

def octahedron(dtype=np.float32):
  verts = np.array([
    (+1,0,0), (0,+1,0), (0,0,+1),
    (-1,0,0), (0,-1,0), (0,0,-1)], dtype=dtype)
  faces = np.array([
    (0,1,2), (1,2,3), (3,2,4), (4,2,0),
    (0,1,5), (1,5,3), (3,5,4), (4,5,0)], dtype=np.int32)
  normals = -np.mean(verts, axis=0) + verts
  normals /= np.linalg.norm(normals, axis=1).reshape(-1,1)
  return verts, faces, normals

def icosahedron(dtype=np.float32):
  p = (1 + np.sqrt(5)) / 2
  verts = np.array([
    (-1,0,p), (1,0,p), (1,0,-p), (-1,0,-p),
    (0,-p,1), (0,p,1), (0,p,-1), (0,-p,-1),
    (-p,-1,0), (p,-1,0), (p,1,0), (-p,1,0)
    ], dtype=dtype)
  faces = np.array([
    (0,1,4), (0,1,5), (1,4,9), (1,9,10), (1,10,5), (0,4,8), (0,8,11), (0,11,5),
    (5,6,11), (5,6,10), (4,7,8), (4,7,9),
    (3,2,6), (3,2,7), (2,6,10), (2,10,9), (2,9,7), (3,6,11), (3,11,8), (3,8,7),
    ], dtype=np.int32)
  normals = -np.mean(verts, axis=0) + verts
  normals /= np.linalg.norm(normals, axis=1).reshape(-1,1)
  return verts, faces, normals

def xyplane(dtype=np.float32, z=0, interleaved=False):
  if interleaved:
    eps = 1e-6
    verts = np.array([
      (-1,-1,z), (-1,1,z), (1,1,z),
      (1-eps,1,z), (1-eps,-1,z), (-1-eps,-1,z)], dtype=dtype)
    faces = np.array([(0,1,2), (3,4,5)], dtype=np.int32)
  else:
    verts = np.array([(-1,-1,z), (-1,1,z), (1,1,z), (1,-1,z)], dtype=dtype)
    faces = np.array([(0,1,2), (0,2,3)], dtype=np.int32)
  normals = np.zeros_like(verts)
  normals[:,2] = -1
  return verts, faces, normals

def mesh_independent_verts(verts, faces, normals=None):
  new_verts = []
  new_normals = []
  for f in faces:
    new_verts.append(verts[f[0]])
    new_verts.append(verts[f[1]])
    new_verts.append(verts[f[2]])
    if normals is not None:
      new_normals.append(normals[f[0]])
      new_normals.append(normals[f[1]])
      new_normals.append(normals[f[2]])
  new_verts = np.array(new_verts)
  new_faces = np.arange(0, faces.size, dtype=faces.dtype).reshape(-1,3)
  if normals is None:
    return new_verts, new_faces
  else:
    new_normals = np.array(new_normals)
    return new_verts, new_faces, new_normals


def stack_mesh(verts, faces):
  n_verts = 0
  mfaces = []
  for idx, f in enumerate(faces):
    mfaces.append(f + n_verts)
    n_verts += verts[idx].shape[0]
  verts = np.vstack(verts)
  faces = np.vstack(mfaces)
  return verts, faces

def normalize_mesh(verts):
  # all the verts have unit distance to the center (0,0,0)
  return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def mesh_triangle_areas(verts, faces):
  a = verts[faces[:,0]]
  b = verts[faces[:,1]]
  c = verts[faces[:,2]]
  x = np.empty_like(a)
  x = a - b
  y = a - c
  t = np.empty_like(a)
  t[:,0] = (x[:,1] * y[:,2] - x[:,2] * y[:,1]);
  t[:,1] = (x[:,2] * y[:,0] - x[:,0] * y[:,2]);
  t[:,2] = (x[:,0] * y[:,1] - x[:,1] * y[:,0]);
  return np.linalg.norm(t, axis=1) / 2

def subdivde_mesh(verts_in, faces_in, n=1):
  for iter in range(n):
    verts = []
    for v in verts_in:
      verts.append(v)
    faces = []
    verts_dict = {}
    for f in faces_in:
      f = np.sort(f)
      i0,i1,i2 = f
      v0,v1,v2 = verts_in[f]

      k = i0*len(verts_in)+i1
      if k in verts_dict:
        i01 = verts_dict[k]
      else:
        i01 = len(verts)
        verts_dict[k] = i01
        v01 = (v0 + v1) / 2
        verts.append(v01)

      k = i0*len(verts_in)+i2
      if k in verts_dict:
        i02 = verts_dict[k]
      else:
        i02 = len(verts)
        verts_dict[k] = i02
        v02 = (v0 + v2) / 2
        verts.append(v02)

      k = i1*len(verts_in)+i2
      if k in verts_dict:
        i12 = verts_dict[k]
      else:
        i12 = len(verts)
        verts_dict[k] = i12
        v12 = (v1 + v2) / 2
        verts.append(v12)

      faces.append((i0,i01,i02))
      faces.append((i01,i1,i12))
      faces.append((i12,i2,i02))
      faces.append((i01,i12,i02))

    verts_in = np.array(verts, dtype=verts_in.dtype)
    faces_in = np.array(faces, dtype=np.int32)
  return verts_in, faces_in


def mesh_adjust_winding_order(verts, faces, normals):
  n0 = normals[faces[:,0]]
  n1 = normals[faces[:,1]]
  n2 = normals[faces[:,2]]
  fnormals = (n0 + n1 + n2) / 3

  v0 = verts[faces[:,0]]
  v1 = verts[faces[:,1]]
  v2 = verts[faces[:,2]]

  e0 = v1 - v0
  e1 = v2 - v0
  fn = np.cross(e0, e1)

  dot = np.sum(fnormals * fn, axis=1)
  ma = dot < 0

  nfaces = faces.copy()
  nfaces[ma,1], nfaces[ma,2] = nfaces[ma,2], nfaces[ma,1]

  return nfaces


def pcl_to_shapecl(verts, colors=None, shape='cube', width=1.0):
  if shape == 'tetrahedron':
    cverts, cfaces, _ = tetrahedron()
  elif shape == 'cube':
    cverts, cfaces, _ = cube()
  elif shape == 'octahedron':
    cverts, cfaces, _ = octahedron()
  elif shape == 'icosahedron':
    cverts, cfaces, _ = icosahedron()
  else:
    raise Exception('invalid shape')

  sverts = np.tile(cverts, (verts.shape[0], 1))
  sverts *= width
  sverts += np.repeat(verts, cverts.shape[0], axis=0)

  sfaces = np.tile(cfaces, (verts.shape[0], 1))
  sfoffset = cverts.shape[0] * np.arange(0, verts.shape[0])
  sfaces += np.repeat(sfoffset, cfaces.shape[0]).reshape(-1,1)

  if colors is not None:
    scolors = np.repeat(colors, cverts.shape[0], axis=0)
  else:
    scolors = None

  return sverts, sfaces, scolors

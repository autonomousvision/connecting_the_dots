import struct
import numpy as np
import collections

def _write_ply_point(fp, x,y,z, color=None, normal=None, binary=False):
  args = [x,y,z]
  if color is not None:
    args += [int(color[0]), int(color[1]), int(color[2])]
  if normal is not None:
    args += [normal[0],normal[1],normal[2]]
  if binary:
    fmt = '<fff'
    if color is not None:
      fmt = fmt + 'BBB'
    if normal is not None:
      fmt = fmt + 'fff'
    fp.write(struct.pack(fmt, *args))
  else:
    fmt = '%f %f %f'
    if color is not None:
      fmt = fmt + ' %d %d %d'
    if normal is not None:
      fmt = fmt + ' %f %f %f'
    fmt += '\n'
    fp.write(fmt % tuple(args))

def _write_ply_triangle(fp, i0,i1,i2, binary):
  if binary:
    fp.write(struct.pack('<Biii', 3,i0,i1,i2))
  else:
    fp.write('3 %d %d %d\n' % (i0,i1,i2))

def _write_ply_header_line(fp, str, binary):
  if binary:
    fp.write(str.encode())
  else:
    fp.write(str)

def write_ply(path, verts, trias=None, color=None, normals=None, binary=False):
  if verts.shape[1] != 3:
    raise Exception('verts has to be of shape Nx3')
  if trias is not None and trias.shape[1] != 3:
    raise Exception('trias has to be of shape Nx3')
  if color is not None and not callable(color) and not isinstance(color, np.ndarray) and color.shape[1] != 3:
    raise Exception('color has to be of shape Nx3 or a callable')

  mode = 'wb' if binary else 'w'
  with open(path, mode) as fp:
    _write_ply_header_line(fp, "ply\n", binary)
    if binary:
      _write_ply_header_line(fp, "format binary_little_endian 1.0\n", binary)
    else:
      _write_ply_header_line(fp, "format ascii 1.0\n", binary)
    _write_ply_header_line(fp, "element vertex %d\n" % (verts.shape[0]), binary)
    _write_ply_header_line(fp, "property float32 x\n", binary)
    _write_ply_header_line(fp, "property float32 y\n", binary)
    _write_ply_header_line(fp, "property float32 z\n", binary)
    if color is not None:
      _write_ply_header_line(fp, "property uchar red\n", binary)
      _write_ply_header_line(fp, "property uchar green\n", binary)
      _write_ply_header_line(fp, "property uchar blue\n", binary)
    if normals is not None:
      _write_ply_header_line(fp, "property float32 nx\n", binary)
      _write_ply_header_line(fp, "property float32 ny\n", binary)
      _write_ply_header_line(fp, "property float32 nz\n", binary)
    if trias is not None:
      _write_ply_header_line(fp, "element face %d\n" % (trias.shape[0]), binary)
      _write_ply_header_line(fp, "property list uchar int32 vertex_indices\n", binary)
    _write_ply_header_line(fp, "end_header\n", binary)

    for vidx, v in enumerate(verts):
      if color is not None:
        if callable(color):
          c = color(vidx)
        elif color.shape[0] > 1:
          c = color[vidx]
        else:
          c = color[0]
      else:
        c = None
      if normals is None:
        n = None
      else:
        n = normals[vidx]
      _write_ply_point(fp, v[0],v[1],v[2], c, n, binary)

    if trias is not None:
      for t in trias:
        _write_ply_triangle(fp, t[0],t[1],t[2], binary)

def faces_to_triangles(faces):
  new_faces = []
  for f in faces:
    if f[0] == 3:
      new_faces.append([f[1], f[2], f[3]])
    elif f[0] == 4:
      new_faces.append([f[1], f[2], f[3]])
      new_faces.append([f[3], f[4], f[1]])
    else:
      raise Exception('unknown face count %d', f[0])
  return new_faces

def read_ply(path):
  with open(path, 'rb') as f:
    # parse header
    line = f.readline().decode().strip()
    if line != 'ply':
      raise Exception('Header error')
    n_verts = 0
    n_faces = 0
    vert_types = {}
    vert_bin_format = []
    vert_bin_len = 0
    vert_bin_cols = 0
    line = f.readline().decode()
    parse_vertex_prop = False
    while line.strip() != 'end_header':
      if 'format' in line:
        if 'ascii' in line:
          binary = False
        elif 'binary_little_endian' in line:
          binary = True
        else:
          raise Exception('invalid ply format')
      if 'element face' in line:
        splits = line.strip().split(' ')
        n_faces = int(splits[-1])
        parse_vertex_prop = False
      if 'element camera' in line:
        parse_vertex_prop = False
      if 'element vertex' in line:
        splits = line.strip().split(' ')
        n_verts = int(splits[-1])
        parse_vertex_prop = True
      if parse_vertex_prop and 'property' in line:
        prop = line.strip().split()
        if prop[1] == 'float':
          vert_bin_format.append('f4')
          vert_bin_len += 4
          vert_bin_cols += 1
        elif prop[1] == 'uchar':
          vert_bin_format.append('B')
          vert_bin_len += 1
          vert_bin_cols += 1
        else:
          raise Exception('invalid property')
        vert_types[prop[2]] = len(vert_types)
      line = f.readline().decode()

    # parse content
    if binary:
      sz = n_verts * vert_bin_len
      fmt = ','.join(vert_bin_format)
      verts = np.ndarray(shape=(1, n_verts), dtype=np.dtype(fmt), buffer=f.read(sz))
      verts = verts[0].astype(vert_bin_cols*'f4,').view(dtype='f4').reshape((n_verts,-1))
      faces = []
      for idx in range(n_faces):
        fmt = '<Biii'
        length = struct.calcsize(fmt)
        dat = f.read(length)
        vals = struct.unpack(fmt, dat)
        faces.append(vals)
      faces = faces_to_triangles(faces)
      faces = np.array(faces, dtype=np.int32)
    else:
      verts = []
      for idx in range(n_verts):
        vals = [float(v) for v in f.readline().decode().strip().split(' ')]
        verts.append(vals)
      verts = np.array(verts, dtype=np.float32)
      faces = []
      for idx in range(n_faces):
        splits = f.readline().decode().strip().split(' ')
        n_face_verts = int(splits[0]) 
        vals = [int(v) for v in splits[0:n_face_verts+1]]
        faces.append(vals)
      faces = faces_to_triangles(faces)
      faces = np.array(faces, dtype=np.int32)

  xyz = None
  if 'x' in vert_types and 'y' in vert_types and 'z' in vert_types:
    xyz = verts[:,[vert_types['x'], vert_types['y'], vert_types['z']]]
  colors = None
  if 'red' in vert_types and 'green' in vert_types and 'blue' in vert_types:
    colors = verts[:,[vert_types['red'], vert_types['green'], vert_types['blue']]]
    colors /= 255
  normals = None
  if 'nx' in vert_types and 'ny' in vert_types and 'nz' in vert_types:
    normals = verts[:,[vert_types['nx'], vert_types['ny'], vert_types['nz']]]

  return xyz, faces, colors, normals


def _read_obj_split_f(s):
  parts = s.split('/')
  vidx = int(parts[0]) - 1
  if len(parts) >= 2 and len(parts[1]) > 0:
    tidx = int(parts[1]) - 1
  else:
    tidx = -1
  if len(parts) >= 3 and len(parts[2]) > 0:
    nidx = int(parts[2]) - 1
  else:
    nidx = -1
  return vidx, tidx, nidx

def read_obj(path):
  with open(path, 'r') as fp:
    lines = fp.readlines()

  verts = []
  colors = []
  fnorms = []
  fnorm_map = collections.defaultdict(list)
  faces = []
  for line in lines:
    line = line.strip()
    if line.startswith('#') or len(line) == 0:
      continue

    parts = line.split()
    if line.startswith('v '):
      parts = parts[1:]
      x,y,z = float(parts[0]), float(parts[1]), float(parts[2])
      if len(parts) == 4 or len(parts) == 7:
        w = float(parts[3])
        x,y,z = x/w, y/w, z/w
      verts.append((x,y,z))
      if len(parts) >= 6:
        r,g,b = float(parts[-3]), float(parts[-2]), float(parts[-1])
        rgb.append((r,g,b))

    elif line.startswith('vn '):
      parts = parts[1:]
      x,y,z = float(parts[0]), float(parts[1]), float(parts[2])
      fnorms.append((x,y,z))

    elif line.startswith('f '):
      parts = parts[1:]
      if len(parts) != 3:
        raise Exception('only triangle meshes supported atm')
      vidx0, tidx0, nidx0 = _read_obj_split_f(parts[0])
      vidx1, tidx1, nidx1 = _read_obj_split_f(parts[1])
      vidx2, tidx2, nidx2 = _read_obj_split_f(parts[2])

      faces.append((vidx0, vidx1, vidx2))
      if nidx0 >= 0:
        fnorm_map[vidx0].append( nidx0 )
      if nidx1 >= 0:
        fnorm_map[vidx1].append( nidx1 )
      if nidx2 >= 0:
        fnorm_map[vidx2].append( nidx2 )

  verts = np.array(verts)
  colors = np.array(colors)
  fnorms = np.array(fnorms)
  faces = np.array(faces)
  
  # face normals to vertex normals
  norms = np.zeros_like(verts)
  for vidx in fnorm_map.keys():
    ind = fnorm_map[vidx]
    norms[vidx] = fnorms[ind].sum(axis=0)
  N = np.linalg.norm(norms, axis=1, keepdims=True)
  np.divide(norms, N, out=norms, where=N != 0)

  return verts, faces, colors, norms

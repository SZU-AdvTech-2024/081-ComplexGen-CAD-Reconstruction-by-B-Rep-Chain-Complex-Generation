import numpy as np
import os

d = 0.2 #curvecorner
d_patch_curve = 0.1
d_patch_corner = 0.2

weight_unary = 10.0
weight_open = 10.0

inf = 0.0

def get_curve_corner_similarity_geom(data, flag_exp = True):
    all_pts = data['corners']['prediction']['position']
    # first point of an edge
    all_e1 = data['curves']['prediction']['points'][:,0:1,:]
    
    # last point of an edge
    all_e2 = data['curves']['prediction']['points'][:,-1:,:]
    all_e1_diff = all_e1 - all_pts
    all_e2_diff = all_e2 - all_pts
    all_e1_dist = np.linalg.norm(all_e1_diff, axis = -1)
    all_e2_dist = np.linalg.norm(all_e2_diff, axis = -1)
    all_curve_corner_dist = np.min(np.array([all_e2_dist, all_e1_dist]),axis = 0)
    # print('all_curve_corner_dist shape: ', all_curve_corner_dist.shape)
    if flag_exp:
      sim = np.exp( -all_curve_corner_dist * all_curve_corner_dist / (d * d))
    else:
      sim = all_curve_corner_dist

    return sim


def get_patch_curve_similarity_geom(data, flag_exp = True):
    all_curve_pts = data['curves']['prediction']['points']
    all_patch_pts = data['patches']['prediction']['points']
    nf = all_patch_pts.shape[0]
    ne = all_curve_pts.shape[0]
    sim = np.zeros([nf, ne])
    for i in range(nf):
      for j in range(ne):
        pts_diff = np.expand_dims(all_curve_pts[j], 1) - all_patch_pts[i]
        # print('pts diff shape: ', pts_diff.shape)
        pts_dist = np.linalg.norm(pts_diff, axis = -1)
        sim[i,j] = np.mean(pts_dist.min(-1))
    if flag_exp:
      sim = np.exp(-sim * sim / (d_patch_curve * d_patch_curve))
  
    return sim


def get_patch_corner_similarity_geom(data, flag_exp = True):
    all_corner_pts = data['corners']['prediction']['position']
    all_patch_pts = data['patches']['prediction']['points']
    nf = all_patch_pts.shape[0]
    nv = all_corner_pts.shape[0]
    sim = np.zeros([nf, nv])
    for i in range(nf):
      for j in range(nv):
        pts_diff = all_patch_pts[i] - all_corner_pts[j]
        pts_dist = np.linalg.norm(pts_diff, axis = -1)
        sim[i,j] = pts_dist.min()
    if flag_exp:
      sim = np.exp(-sim * sim / (d_patch_corner * d_patch_corner))
    return sim
    
    
def construct_vertices():
    return np.array([
        [0.0,0.5,1.0],
        [0.0,0.5,-1.0],
        [-0.5,-0.5,1.0],
        [-0.5,-0.5,-1.0],
        [0.5,-0.5,1.0],
        [0.5,-0.5,-1.0],
    ])


def construct_edges(vertices):
    vertex_pairs = [
        [0, 2],
        [0, 4],
        [2, 4],
        [1, 3],
        [1, 5],
        [3, 5],
        [0, 1],
        [2, 3],
        [4, 5]
    ]
    edges = []
    for pairs in vertex_pairs:
        step = (vertices[pairs[0]] - vertices[pairs[1]]) / 15.0
        edge = []
        for i in range(16):
            edge.append(vertices[pairs[1]] + i * step)
        edges.append(edge)
    return np.array(edges)
        
  
def construct_faces(edges):
    edge_pairs = [
        [0, 3],
        [1, 4],
        [2, 5]
    ]
    
    faces = []
    for pair in edge_pairs:
        step = (edges[pair[0]] - edges[pair[1]]) / 15.0
        face = []
        for i in range(16):
            face.append(edges[pair[1]] + i * step)
        faces.append(np.array(face).reshape([-1, 3]))
    return np.array(faces)


def construct_data():
    data = dict()
    data['corners'] = dict()
    data['curves'] = dict()
    data['patches'] = dict()
    data['corners']['prediction'] = dict()
    data['curves']['prediction'] = dict()
    data['patches']['prediction'] = dict()
    
    # Position
    data['corners']['prediction']['position'] = construct_vertices()
    data['curves']['prediction']['points'] = construct_edges(data['corners']['prediction']['position'])
    data['patches']['prediction']['points'] = construct_faces(data['curves']['prediction']['points'])
    
    # Openness
    data['curves']['prediction']['closed_prob'] = np.zeros([len(data['curves']['prediction']['points'])])
    data['patches']['prediction']['closed_prob'] = np.zeros([len(data['patches']['prediction']['points'])])
    
    # Validity
    data['corners']['prediction']['valid_prob'] = np.ones([len(data['corners']['prediction']['position'])])
    data['curves']['prediction']['valid_prob'] = np.ones([len(data['curves']['prediction']['points'])])
    data['patches']['prediction']['valid_prob'] = np.ones([len(data['patches']['prediction']['points'])])
    
    data['patch_curve_similarity'] = np.zeros([data['patches']['prediction']['points'].shape[0], data['curves']['prediction']['points'].shape[0]])
    data['curve_corner_similarity'] = np.zeros([data['curves']['prediction']['points'].shape[0], data['corners']['prediction']['position'].shape[0]])
    data['patch_corner_similarity'] = np.zeros([data['patches']['prediction']['points'].shape[0], data['corners']['prediction']['position'].shape[0]])
    
    data['patch_curve_similarity'][0][0] = 1.0
    data['patch_curve_similarity'][0][3] = 1.0
    data['patch_curve_similarity'][0][6] = 1.0
    data['patch_curve_similarity'][0][7] = 1.0
    data['patch_curve_similarity'][1][1] = 1.0
    data['patch_curve_similarity'][1][4] = 1.0
    data['patch_curve_similarity'][1][6] = 1.0
    data['patch_curve_similarity'][1][8] = 1.0
    data['patch_curve_similarity'][2][2] = 1.0
    data['patch_curve_similarity'][2][5] = 1.0
    data['patch_curve_similarity'][2][7] = 1.0
    data['patch_curve_similarity'][2][8] = 1.0
    
    data['curve_corner_similarity'][0][0] = 1.0
    data['curve_corner_similarity'][0][2] = 1.0
    data['curve_corner_similarity'][1][0] = 1.0
    data['curve_corner_similarity'][1][4] = 1.0
    data['curve_corner_similarity'][2][2] = 1.0
    data['curve_corner_similarity'][2][4] = 1.0
    data['curve_corner_similarity'][3][1] = 1.0
    data['curve_corner_similarity'][3][3] = 1.0
    data['curve_corner_similarity'][4][1] = 1.0
    data['curve_corner_similarity'][4][5] = 1.0
    data['curve_corner_similarity'][5][3] = 1.0
    data['curve_corner_similarity'][5][5] = 1.0
    data['curve_corner_similarity'][6][0] = 1.0
    data['curve_corner_similarity'][6][1] = 1.0
    data['curve_corner_similarity'][7][2] = 1.0
    data['curve_corner_similarity'][7][3] = 1.0
    data['curve_corner_similarity'][8][4] = 1.0
    data['curve_corner_similarity'][8][5] = 1.0
    
    data['patch_corner_similarity'][0][0] = 1.0
    data['patch_corner_similarity'][0][1] = 1.0
    data['patch_corner_similarity'][0][2] = 1.0
    data['patch_corner_similarity'][0][3] = 1.0
    data['patch_corner_similarity'][1][0] = 1.0
    data['patch_corner_similarity'][1][1] = 1.0
    data['patch_corner_similarity'][1][4] = 1.0
    data['patch_corner_similarity'][1][5] = 1.0
    data['patch_corner_similarity'][2][2] = 1.0
    data['patch_corner_similarity'][2][3] = 1.0
    data['patch_corner_similarity'][2][4] = 1.0
    data['patch_corner_similarity'][2][5] = 1.0
    return data


def get_valid_edge_indices(edge_face_connectivity):  
  valid_edge_indices = []
  face_intersections = []
  invalid_edge_indices = []
  same_edge = dict()
  redundant_edge_indices_to_edge_indices = dict()
  for connectivity in edge_face_connectivity:
    intersection = frozenset([connectivity[1], connectivity[2]])
    if len(intersection) == 2:
      if intersection not in face_intersections:
        face_intersections.append(intersection)
        valid_edge_indices.append(connectivity[0])
        same_edge[intersection] = connectivity[0]
      else:
        redundant_edge_indices_to_edge_indices[connectivity[0]] = same_edge[intersection]
    elif len(intersection) == 1:
      invalid_edge_indices.append(connectivity[0])
  return valid_edge_indices, redundant_edge_indices_to_edge_indices, invalid_edge_indices

def GetExpDataPath(folder_path=None):
  
  if folder_path is None:
    return [str(os.path.join(folder_path, f)) for f in os.listdir("./") if f.endswith('.npz') and not f.endswith('_optimized.npz')]
  else:
    return [str(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith('.npz') and not f.endswith('_optimized.npz')]
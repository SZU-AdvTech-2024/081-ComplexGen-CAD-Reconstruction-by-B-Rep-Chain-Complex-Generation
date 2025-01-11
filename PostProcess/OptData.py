import numpy as np
import os
import utils
import trimesh

class OptData:
    """
    This Class is used to load data of this experiments
    
    - `pred` is to represent the data comes from prediction data
    - `gt` is to represent the data comes from ground truth data
    
    - Attributes:
        - valid (bool): The flag to indicate whether data is succuessfully loaded
        - file_open (bool): The flag to indicate whether file can be opened
        - file_path (str): Data file path
        - tolerance (float): The tolerance of the accuracy
        - fileNumber (str): The number of this data
        - connectivityIsCorrect (bool): Whether the predicted data is definitely to have errors
        
        - predFaces (NDArray[np.float64]): Sample points positions of all predicted faces (face number, 16, 16, 3)
        - predEdges (NDArray[np.float64]): Sample points positions of all predicted edges (edge number, 16, 3)
        - predVertices (NDArray[np.float64]): Positions of all predicted faces (vertex number, 3)
        - predEdgeOpen (NDArray[np.float64]): Openness of all edges (edge number, )
        - predEdgeFace (NDArray[np.int64]): An edge is formed by the intersection of which two faces (n, 3)
        - predFE (NDArray[np.float32]): Face-Edge connectivity (face number, edge number)
        - predFV (NDArray[np.float32]): Face-Vertex connectivity (face number, vertex number)
        - predEV (NDArray[np.float32]): Edge-Vertex connectivity (edge number, vertex number)
        
        - gtFaces (NDArray[np.float64]): Sample points positions of all predicted faces (face number, 16, 16, 3)
        - gtEdges (NDArray[np.float64]): Sample points positions of all predicted edges (edge number, 16, 3)
        - gtVertices (NDArray[np.float64]): Positions of all predicted faces (vertex number, 3)
        - gtEdgeOpen (NDArray[np.float64]): Openness of all edges (edge number, )
        - gtEdgeFace (NDArray[np.int64]): An edge is formed by the intersection of which two faces (n, 3)
        - gtFE (NDArray[np.float32]): Face-Edge connectivity (face number, edge number)
        - gtFV (NDArray[np.float32]): Face-Vertex connectivity (face number, vertex number)
        - gtEV (NDArray[np.float32]): Edge-Vertex connectivity (edge number, vertex number)
    """
    def __GetValidEdgeIndex(self, edgeFace):  
        valid_edge_indices = []
        face_intersections = []
        invalid_edge_indices = []
        same_edge = dict()
        redundant_edge_indices_to_edge_indices = dict()
        for connectivity in edgeFace:
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


    def __GetEdgeOpenness(self, data):
        dist = np.linalg.norm(data[:, 0] - data[:, -1], axis=1) 
        compare_result = dist < self.tolerance
        return compare_result.astype(np.float64)
    
    
    def __GetValidEdgeFaceIntersections(self, edgeFace, validEdgeIndex, edgeIndexReplacementDict):
        result = []
        for edgeFace1Face2 in edgeFace:
            if edgeFace1Face2[0] in validEdgeIndex:
                result.append([validEdgeIndex.index(edgeFace1Face2[0]), edgeFace1Face2[1], edgeFace1Face2[2]])
            elif edgeFace1Face2[0] in edgeIndexReplacementDict.keys():
                result.append([validEdgeIndex.index(edgeIndexReplacementDict[edgeFace1Face2[0]]), edgeFace1Face2[1], edgeFace1Face2[2]])
        return np.array(result, dtype=np.int64)
            
    
    def __GetDynamicTolerance(self, edges, edgeOpenness):
        minDistance = []
        for i, edge in enumerate(edges):
            endPoint1 = edge[0]
            endPoint2 = edge[-1]
            distance = []
            for j, _edge in enumerate(edges):
                if edgeOpenness[j] == 0.0 or i == j:
                    continue
                distance.append(np.linalg.norm(endPoint1 - _edge[0]))
                distance.append(np.linalg.norm(endPoint2 - _edge[0]))
                distance.append(np.linalg.norm(endPoint1 - _edge[-1]))
                distance.append(np.linalg.norm(endPoint2 - _edge[-1]))
            minDistance += distance
        
        minDistance.sort()
        minDistance = minDistance[:2 * 4 * edges.shape[0]] 
        tolerance = np.sum(minDistance, dtype=np.float64) / (2.0 * 4.0 * len(edges))
        return tolerance
     
    def __GetVertices(self, edges, edgeOpenness, bias=1e-3):
        vertices = []
        EdgeVertexConnectivityIndex = [] # element: [edge index, vertex index]
        
        def ExistSimilarVertex(v, vertices, tolerance):
            for vertex in vertices:
                if np.linalg.norm(v - vertex) < tolerance:
                    return True
            return False
        
        minDistance = []
        for i, edge in enumerate(edges):
            endPoint1 = edge[0]
            endPoint2 = edge[-1]
            distance = []
            for j, _edge in enumerate(edges):
                if edgeOpenness[j] == 0.0 or i == j:
                    continue
                distance.append(np.linalg.norm(endPoint1 - _edge[0]))
                distance.append(np.linalg.norm(endPoint2 - _edge[0]))
                distance.append(np.linalg.norm(endPoint1 - _edge[-1]))
                distance.append(np.linalg.norm(endPoint2 - _edge[-1]))
            minDistance += distance
        
        minDistance.sort()
        minDistance = minDistance[:2 * 4 * edges.shape[0]] 
        tolerance = np.sum(minDistance, dtype=np.float64) / (2.0 * 4.0 * len(edges))
        
        vertices = []
        for i, edge in enumerate(edges):
            if edgeOpenness[i] == 0.0:
                continue
            endPoint1 = edge[0]
            endPoint2 = edge[-1]
            
            if not ExistSimilarVertex(endPoint1, vertices, tolerance):
                EdgeVertexConnectivityIndex.append([i, len(vertices)])
                vertices.append(endPoint1)
            else:
                index = -1
                for j, vertex in enumerate(vertices):
                    if np.linalg.norm(endPoint1 - vertex) < tolerance:
                        index = j
                        break
                EdgeVertexConnectivityIndex.append([i, index])
                
            if not ExistSimilarVertex(endPoint2, vertices, tolerance):
                EdgeVertexConnectivityIndex.append([i, len(vertices)])
                vertices.append(endPoint2)
            else:
                index = -1
                for j, vertex in enumerate(vertices):
                    if np.linalg.norm(endPoint2 - vertex) < tolerance:
                        index = j
                        break
                EdgeVertexConnectivityIndex.append([i, index])
        
        return np.array(vertices, dtype=np.float64), EdgeVertexConnectivityIndex
         
    def ___GetVertices(self, v_edges, v_edge_face_connectivity, edgeOpenness):
        # Find the 3-node ring of the face
        num_faces = self.predFaces.shape[0]
        num_edges = self.predEdges.shape[0]
        face_adj = np.zeros((num_faces, num_faces), dtype=bool)
        face_adj[v_edge_face_connectivity[:, 1], v_edge_face_connectivity[:, 2]] = True 
        inv_edge_face_connectivity = {}
        for i, (edge, face1, face2) in enumerate(v_edge_face_connectivity):
            inv_edge_face_connectivity[(face1, face2)] = edge   
        
        vertices = []
        cached_vertices = []   
        EVConnectivityIndex = [] 
        FVConnectivityIndex = [] 
        for face1 in range(num_faces):
            for face2 in range(num_faces):
                if not face_adj[face1, face2]:
                    continue
                for face3 in range(num_faces):
                    if not face_adj[face1, face3]:
                        continue
                    if not face_adj[face2, face3]:
                        continue    
                    line1 = self.predEdges[inv_edge_face_connectivity[(face1, face2)]]
                    line2 = self.predEdges[inv_edge_face_connectivity[(face1, face3)]]
                    line3 = self.predEdges[inv_edge_face_connectivity[(face2, face3)]]  
                    key = (set([face1, face2, face3]), set([inv_edge_face_connectivity[(face1, face2)], inv_edge_face_connectivity[(face1, face3)], inv_edge_face_connectivity[(face2, face3)]]))
                    if key in cached_vertices:
                        continue    
                    def dis(a, b):
                        return np.linalg.norm(a - b)    
                    tolerence = 1e-1
                    if dis(line1[0], line2[0]) < tolerence and dis(line1[0], line3[0]) < tolerence and dis(line2[0], line3[0]) < tolerence:
                        FVConnectivityIndex.append([list(key[0])[0], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[1], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[2], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[0], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[1], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[2], len(vertices)])
                        cached_vertices.append(key)
                        vertices.append((line1[0] + line2[0] + line3[0]) / 3)
                    elif dis(line1[0], line2[0]) < tolerence and dis(line1[0], line3[-1]) < tolerence and dis(line2[0], line3[-1]) < tolerence:
                        FVConnectivityIndex.append([list(key[0])[0], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[1], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[2], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[0], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[1], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[2], len(vertices)])
                        cached_vertices.append(key)
                        vertices.append((line1[0] + line2[0] + line3[-1]) / 3)
                    elif dis(line1[0], line2[-1]) < tolerence and dis(line1[0], line3[0]) < tolerence and dis(line2[-1], line3[0]) < tolerence:
                        FVConnectivityIndex.append([list(key[0])[0], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[1], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[2], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[0], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[1], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[2], len(vertices)])
                        cached_vertices.append(key)
                        vertices.append((line1[0] + line2[-1] + line3[0]) / 3)
                    elif dis(line1[0], line2[-1]) < tolerence and dis(line1[0], line3[-1]) < tolerence and dis(line2[-1], line3[-1]) < tolerence:
                        FVConnectivityIndex.append([list(key[0])[0], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[1], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[2], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[0], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[1], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[2], len(vertices)])
                        cached_vertices.append(key)
                        vertices.append((line1[0] + line2[-1] + line3[-1]) / 3)
                    elif dis(line1[-1], line2[0]) < tolerence and dis(line1[-1], line3[0]) < tolerence and dis(line2[0], line3[0]) < tolerence:
                        FVConnectivityIndex.append([list(key[0])[0], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[1], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[2], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[0], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[1], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[2], len(vertices)])
                        cached_vertices.append(key)
                        vertices.append((line1[-1] + line2[0] + line3[0]) / 3)
                    elif dis(line1[-1], line2[0]) < tolerence and dis(line1[-1], line3[-1]) < tolerence and dis(line2[0], line3[-1]) < tolerence:
                        FVConnectivityIndex.append([list(key[0])[0], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[1], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[2], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[0], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[1], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[2], len(vertices)])
                        cached_vertices.append(key)
                        vertices.append((line1[-1] + line2[0] + line3[-1]) / 3)
                    elif dis(line1[-1], line2[-1]) < tolerence and dis(line1[-1], line3[0]) < tolerence and dis(line2[-1], line3[0]) < tolerence:
                        FVConnectivityIndex.append([list(key[0])[0], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[1], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[2], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[0], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[1], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[2], len(vertices)])
                        cached_vertices.append(key)
                        vertices.append((line1[-1] + line2[-1] + line3[0]) / 3)
                    elif dis(line1[-1], line2[-1]) < tolerence and dis(line1[-1], line3[-1]) < tolerence and dis(line2[-1], line3[-1]) < tolerence:
                        FVConnectivityIndex.append([list(key[0])[0], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[1], len(vertices)])
                        FVConnectivityIndex.append([list(key[0])[2], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[0], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[1], len(vertices)])
                        EVConnectivityIndex.append([list(key[1])[2], len(vertices)])
                        cached_vertices.append(key)
                        vertices.append((line1[-1] + line2[-1] + line3[-1]) / 3)
                    else:
                        continue
        return np.array(vertices,dtype=np.float64), FVConnectivityIndex, EVConnectivityIndex
            
    def __WriteEdgePLY(self, indices=None):
        """
        Debug Only
        """
        if indices is None:
            trimesh.PointCloud(self.predEdges.reshape([-1, 3])).export("./experiments/prediction_output/93_edges.ply")
        else:
            for index in indices:
                trimesh.PointCloud(self.predEdges[index]).export(f"./experiments/prediction_output/93_{index}_edge.ply")
            trimesh.PointCloud(self.predEdges[indices].reshape([-1, 3])).export("./experiments/prediction_output/93_compound_edges.ply")
    
    
    def __GetEdgeProb(self):
        edgeProb = np.zeros([self.predEdges.shape[0],], dtype=np.float64)
        for edgeFace1Face2 in self.predEdgeFace:
            edgeProb[edgeFace1Face2[0]] += (self.faceAdjProb[edgeFace1Face2[1]][edgeFace1Face2[2]])
        return edgeProb / 2.0
    
        
    def __ProcessData(self):
        self.connectivityIsCorrect = len(self.gtEdgeFace) == len(self.predEdgeFace)
        
        # Prediction
        ## Edges
        validEdgeIndex, redundant_edge_indices_To_edge_indices, _ = self.__GetValidEdgeIndex(self.predEdgeFace) # Get valid edge index and its replacement first
        self.predEdges = self.predEdges[validEdgeIndex] # Get edge sample points
        self.predEdgeOpenness = 1.0 - self.__GetEdgeOpenness(self.predEdges) # Get EdgeOpenness
        self.predEdgeFace = self.__GetValidEdgeFaceIntersections(self.predEdgeFace, validEdgeIndex, redundant_edge_indices_To_edge_indices)
        self.predEdgesProb = self.__GetEdgeProb() # Get the valid probability of every edge
        # With valid edge-face connectivity, creating the FE connectivity matrix
        self.predFE = np.zeros([len(self.predFaces), len(self.predEdges)], dtype=np.float64) # Get
        for edgeFace in self.predEdgeFace:
            self.predFE[edgeFace[1]][edgeFace[0]] = 1.0
            self.predFE[edgeFace[2]][edgeFace[0]] = 1.0

        # self.predVertices, EdgeVertexConnectivityIndex = self.__GetVertices(self.predEdges, self.predEdgeOpenness, 2e-2)
        self.predVertices, FaceVertexConnectivityIndex, EdgeVertexConnectivityIndex = self.___GetVertices(self.predEdges, self.predEdgeFace, self.predEdgeOpenness)
        
        self.predEV = np.zeros([len(self.predEdges), len(self.predVertices)], dtype=np.float64)
        for index in EdgeVertexConnectivityIndex:
            self.predEV[index[0]][index[1]] = 0.25
        
        self.predFV = np.zeros([len(self.predFaces), len(self.predVertices)], dtype=np.float64)
        for index in FaceVertexConnectivityIndex:
            self.predFV[index[0]][index[1]] = 1.0
            
        # self.predFV = np.matmul(self.predFE, self.predEV, dtype=np.float64)
        
        # GT
        validEdgeIndex, redundant_edge_indices_To_edge_indices, _ = self.__GetValidEdgeIndex(self.gtEdgeFace)
        self.gtEdges = self.gtEdges[validEdgeIndex]
        self.gtEdgeOpenness = 1.0 - self.__GetEdgeOpenness(self.gtEdges)

        self.gtFE = np.zeros([len(self.gtFaces), len(self.gtEdges)], dtype=np.float64)
        self.gtEdgeFace = self.__GetValidEdgeFaceIntersections(self.gtEdgeFace, validEdgeIndex, redundant_edge_indices_To_edge_indices)
        for edgeFace in self.gtEdgeFace:
            self.gtFE[edgeFace[1]][edgeFace[0]] = 0.5
            self.gtFE[edgeFace[2]][edgeFace[0]] = 0.5
        
            
    def __init__(self, filepath, sampleFrequence=16, optGeomWeight=0.5, tolerance=2e-2):
        self.valid = True
        self.file_open = True
        
        self.filePath = filepath
        try:
            data = np.load(filepath)
        except:
            self.valid = False
            self.file_open = False
            return
        self.tolerance = tolerance
        self.sampleFrequence = sampleFrequence
        self.optGeomWeight = optGeomWeight
        self.fileNumber = filepath.split('/')[-1].split('.')[0]
        
        self.predFaces = data['pred_face'].reshape([-1, sampleFrequence * sampleFrequence, 3]).astype(np.float64)
        self.faceAdjProb = data['pred_face_adj_prob'].reshape([self.predFaces.shape[0], self.predFaces.shape[0]]).astype(np.float64)
        self.predEdges = data['pred_edge'].astype(np.float64)
        self.predEdgeOpenness = []
        self.predEdgeFace = data['pred_edge_face_connectivity'] # [Edge index, Face1 index, Face2 index]
        self.predVertices = []
        self.predFE = None
        self.predFV = None
        self.predEV = None
        
        # No use for now
        self.gtFaces = data['gt_face'].astype(np.float64)
        self.gtEdges = data['gt_edge'].astype(np.float64)
        self.gtVertices = []
        self.gtEdgeOpenness = []
        self.gtEdgeFace = data['gt_edge_face_connectivity'] # [Edge index, Face1 index, Face2 index]
        self.gtFE = None
        self.gtFV = None
        self.gtEV = None
        
        del data
        self.__ProcessData()
        # try:
        #     self.__ProcessData()
        # except:
        #     self.valid = False
    
    
    def __GetEVGeom(self, vertices, edges, flag_exp = True):
        all_pts = vertices
        # first point of an edge
        all_e1 = edges[:,0:1,:]

        # last point of an edge
        all_e2 = edges[:,-1:,:]
        all_e1_diff = all_e1 - all_pts
        all_e2_diff = all_e2 - all_pts
        all_e1_dist = np.linalg.norm(all_e1_diff, axis = -1)
        all_e2_dist = np.linalg.norm(all_e2_diff, axis = -1)
        all_curve_corner_dist = np.min(np.array([all_e2_dist, all_e1_dist]),axis = 0)
        # print('all_curve_corner_dist shape: ', all_curve_corner_dist.shape)
        if flag_exp:
          sim = np.exp( -all_curve_corner_dist * all_curve_corner_dist / (utils.d * utils.d))
        else:
          sim = all_curve_corner_dist

        return sim.astype(np.float64)
    
    
    def __GetFVGeom(self, vertices, faces, flag_exp = True):
        all_corner_pts = vertices
        all_patch_pts = faces
        nf = all_patch_pts.shape[0]
        nv = all_corner_pts.shape[0]
        sim = np.zeros([nf, nv])
        for i in range(nf):
          for j in range(nv):
            pts_diff = all_patch_pts[i] - all_corner_pts[j]
            pts_dist = np.linalg.norm(pts_diff, axis = -1)
            sim[i,j] = pts_dist.min()
        if flag_exp:
          sim = np.exp(-sim * sim / (utils.d_patch_corner * utils.d_patch_corner))
        return sim.astype(np.float64)
    
    
    def __GetFEGeom(self, edges, faces, flag_exp = True):
        all_curve_pts = edges
        all_patch_pts = faces
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
          sim = np.exp(-sim * sim / (utils.d_patch_curve * utils.d_patch_curve))
    
        return sim.astype(np.float64)
    
    
    def GetPredCoefficient(self):
        if not self.valid:
            return None
        edgeVertexSimilarityGeom = np.array([], dtype=np.float64)
        faceVerticesSimilarityGeom = np.array([], dtype=np.float64)
        faceEdgeSimilarityGeom = np.array([], dtype=np.float64)
        if self.predVertices.shape[0] > 0:
            edgeVertexSimilarityGeom = self.__GetEVGeom(self.predVertices, self.predEdges)
            faceVerticesSimilarityGeom = self.__GetFVGeom(self.predVertices, self.predFaces)
        faceEdgeSimilarityGeom = self.__GetFEGeom(self.predEdges, self.predFaces)
        
        
        edges = np.ones([len(self.predEdges)], dtype=np.float64) - 0.5
        # if self.fileNumber == '00000093':
            # edges[7] = 0.25
        
        return np.concatenate([
            np.ones([len(self.predVertices)], dtype=np.float64) - 0.5,
            # self.predEdgesProb,
            edges,
            np.ones([len(self.predFaces)], dtype=np.float64),
            (self.optGeomWeight * edgeVertexSimilarityGeom + (1.0 - self.optGeomWeight) * self.predEV).reshape([-1]),
            (self.optGeomWeight * faceVerticesSimilarityGeom + (1.0 - self.optGeomWeight) * self.predFV).reshape([-1]),
            (self.optGeomWeight * faceEdgeSimilarityGeom + (1.0 - self.optGeomWeight) * self.predFE).reshape([-1]),
            self.predEdgeOpenness
        ])

if __name__ == '__main__':
    data = OptData('./experiments/default/ckpt/00000093(1).npz', 16, 1)

    # data = np.load('./experiments/prediction_output/00000093_optimized.npz')
            
    # trimesh.PointCloud(data.predEdges[1]).export('./experiments/default/ckpt/edge1.ply')
    # trimesh.PointCloud(data.predEdges[7]).export('./experiments/default/ckpt/edge7.ply')
    # trimesh.PointCloud(data.predEdges[8]).export('./experiments/default/ckpt/edge8.ply')
    # trimesh.PointCloud(data.predEdges[10]).export('./experiments/default/ckpt/edge10.ply')
    # trimesh.PointCloud(data.predVertices[2].reshape([1, 3])).export('./experiments/default/ckpt/vertex2.ply')
    # trimesh.PointCloud(data.predVertices[3].reshape([1, 3])).export('./experiments/default/ckpt/vertex3.ply')
    # trimesh.PointCloud(data.predVertices[6].reshape([1, 3])).export('./experiments/default/ckpt/vertex6.ply')
    # trimesh.PointCloud(data.predVertices[7].reshape([1, 3])).export('./experiments/default/ckpt/vertex7.ply')
    
    # trimesh.PointCloud(data.predFaces[0]).export('./experiments/default/ckpt/face0.ply')
    # trimesh.PointCloud(data.predFaces[2]).export('./experiments/default/ckpt/face2.ply')
    # trimesh.PointCloud(data.predFaces[8]).export('./experiments/default/ckpt/face8.ply')
    # trimesh.PointCloud(data.predFaces.reshape([-1, 3])).export('./experiments/default/ckpt/face.ply')
    # trimesh.PointCloud(data.predFaces[8]).export('./experiments/default/ckpt/debug3.ply')
    for i in range(7):
        trimesh.PointCloud(data.predFaces[i].reshape([-1, 3])).export('./experiments/default/ckpt/faces{}.ply'.format(i))
    print(data.GetPredCoefficient())
    if data is None:
        print("Error happened when processing data")
    

    
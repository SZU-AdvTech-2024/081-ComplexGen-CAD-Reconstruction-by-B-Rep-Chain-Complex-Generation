import gurobipy as gp
from gurobipy import GRB
import numpy as np
from OptData import OptData
import argparse
from pathlib import Path

from utils import *

max_time = 5*60

options = {
    "WLSACCESSID": "65efab4a-0464-4cdb-aeba-30565a53583b",
    "WLSSECRET": "ea97b187-f36a-4aec-beb8-b672f7568f76",
    "LICENSEID": 2538923,
}

gurobi_file_opened = False

env = gp.Env(params=options)
itr = 0

def gurobi_optimize(c, nv, ne, nf):
    ###############################
    # Build Gurobi Optimize model #
    ###############################
    m = gp.Model("binaryprogramming", env=env)
    m.setParam('TimeLimit', max_time)
    m.setParam('Method', 3)
    m.setParam('Threads', 2)  
    
    vars = []
    for i in range(c.shape[0]):
        vars.append(m.addVar(vtype=GRB.BINARY))
        
    obj = None
    flag_none = True
    for i in range(c.shape[0]):
        if flag_none:
            flag_none = False
            obj = vars[i] * c[i]
        else:
            obj = obj + vars[i] * c[i]
    m.setObjective(obj, GRB.MINIMIZE)
    
    ###########################
    # \sum_j FE[i, j] = 2E[i] #
    ###########################
    for edge_index in range(ne):
        faceEdgeConstraints = 2 * vars[nv + edge_index]
        for face_index in range(nf):
            faceEdgeConstraints += (-1 * vars[nv + ne + nf + nv * ne + nv * nf + face_index * ne + edge_index]) 
        m.addConstr(faceEdgeConstraints == 0)
        
    ###############################
    # \sum_j EV[i, j] = 2E[i]O[i] #
    ###############################
    for edge_index in range(ne):
        edgeVertexConstraints = 2 * vars[nv + edge_index] * vars[ne + nv + nf + nv * ne + nf * ne + nf * nv + edge_index]
        for vertex_index in range(nv):
            edgeVertexConstraints += (-1 * vars[nv + ne + nf + edge_index * nv + vertex_index])
        m.addQConstr(edgeVertexConstraints == 0)
    
    ######################
    # FE \times EV = 2FV #
    ######################
    for face_index in range(nf):
        for vertex_index in range(nv):
            faceLoopConstraints = 2 * vars[nv + ne + nf + nv * ne + face_index * nv + vertex_index]
            # Set all FE[i,k] EV[k, j] to 1
            for k in range(ne):
                # FE[i, k] EV[k, j]
                faceLoopConstraints += (-1 * vars[nv + ne + nf + nv * ne + nv * nf + face_index * ne + k] * vars[nv + ne + nf + k * nv + vertex_index])
            m.addQConstr(faceLoopConstraints == 0)
    
    ###################
    # FE[i,j] <= F[i] #
    ###################
    for face_index in range(nf):
        for edge_index in range(ne):
            m.addConstr((vars[nv + ne + nf + nv * ne + nf * nv + face_index * ne + edge_index] - vars[nv + ne + face_index]) <= 0 )
    
    ##########################
    # F[i] <= \sum_j FE[i,j] #
    ##########################
    if nv > 0:
        for face_index in range(nf):
            constraint = vars[nv + ne + face_index]
            for edge_index in range(ne):
                constraint += (-1 * vars[nv + ne + nf + nv * ne + nf * nv + face_index * ne + edge_index])
            m.addConstr(constraint <= 0)
    
    ###################
    # EV[i,j] <= V[i] #
    ###################
    for edge_index in range(ne):
        for vertex_index in range(nv):
            m.addConstr((vars[nv + ne + nf + edge_index * nv + vertex_index] - vars[vertex_index]) <= 0)
    
    ###################
    # v[j] <= \sum_k EV[k,j] #
    ###################
    for vertex_index in range(nv):
        constraint = vars[vertex_index]
        for edge_index in range(ne):
            constraint += (-1 * vars[nv + ne + nf + edge_index * nv + vertex_index])
        m.addConstr(constraint <= 0)
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            Current_Objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            print(Current_Objective)
            vars = model.getVars()
            vertex = np.array([])
            edge = np.array([])
            face = np.array([])
            EdgeVertex = np.array([])
            FaceVertex = np.array([])
            FaceEdge = np.array([])
            Openness = np.array([])
            Edges_ = np.array([])
            for i, var in enumerate(vars):
                value = model.cbGetSolution(var)
                if i < nv:
                    vertex = np.append(vertex, value)  # Reassign the result
                elif i >= nv and i < nv + ne:
                    Edges_ = np.append(Edges_, value)  # Reassign the result
                    edge = np.append(edge, value)  # Reassign the result
                elif i >= nv + ne and i < nv + ne + nf:
                    face = np.append(face, value)  # Reassign the result
                elif i >= nv + ne + nf and i < nv + ne + nf + ne * nv:
                    EdgeVertex = np.append(EdgeVertex, value)  # Reassign the result
                elif i >= nv + ne + nf + ne * nv and i < nv + ne + nf + ne * nv + nf * nv:
                    FaceVertex = np.append(FaceVertex, value)  # Reassign the result
                elif i >= nv + ne + nf + ne * nv + nf * nv and i < nv + ne + nf + ne * nv + nf * nv + nf * ne:
                    FaceEdge = np.append(FaceEdge, value)  # Reassign the result
                elif i >= nv + ne + nf + ne * nv + nf * nv + nf * ne and i < nv + ne + nf + ne * nv + nf * nv + nf * ne + ne:
                    Openness = np.append(Openness, value)  # Reassign the result
            # print(vertex * c[:nv])
            # print(edge * c[nv:nv + ne])
            # print(face * c[nv + ne:nv + ne + nf])
            # print(EdgeVertex * c[nv + ne + nf:nv + ne + nf + ne * nv])
            # print(FaceVertex * c[nv + ne + nf + ne * nv:nv + ne + nf + ne * nv + nv * nf])
            # print(FaceEdge * c[nv + ne + nf + ne * nv + nv * nf:nv + ne + nf + ne * nv + nv * nf + nf * ne])
            # print(Openness * c[nv + ne + nf + ne * nv + nv * nf + nf * ne:nv + ne + nf + ne * nv + nv * nf + nf * ne + ne])
    
    m.optimize(callback)
    
    x = []
    for v in m.getVars():
        x.append(v.x)
    
    vertex_valid_prob = np.round(np.array(x[:nv]))
    edge_valid_prob = np.round(np.array(x[nv:nv + ne]))
    face_valid_prob = np.round(np.array(x[nv + ne:nv + ne + nf]))
    ev_connectivity = np.round(np.array(x[nv + ne + nf:nv + ne + nf + ne * nv]))
    fv_connectivity = np.round(np.array(x[nv + ne + nf + ne * nv:nv + ne + nf + ne * nv + nv * nf]))
    fe_connectivity = np.round(np.array(x[nv + ne + nf + ne * nv + nv * nf:nv + ne + nf + ne * nv + nv * nf + nf * ne]))
    openness = np.round(np.array(x[nv + ne + nf + ne * nv + nv * nf + nf * ne:nv + ne + nf + ne * nv + nv * nf + nf * ne + ne]))

    return vertex_valid_prob, edge_valid_prob, face_valid_prob, ev_connectivity, fv_connectivity, fe_connectivity, openness

if __name__ == '__main__':
    
    # data = np.load('./experiments/prediction_output/00000093.npz')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folders", type=str, required=True, help="Path to the model folders")
    args = parser.parse_args()
    
    folders = os.listdir(args.model_folders)
    
    for folder in folders:
        folder = Path(folder)
        file_to_extract = None
        for file_name in os.listdir(folder):
            file_name = Path(file_name)
            if file_name.suffix == '.npz':
                file_to_extract = str(file_name)
                break
        
        if file_to_extract is None:
            continue
        
        saving_dir = folder.parent
        stem = folder.stem
        
        data = OptData(file_to_extract, 16, 0.5)
        coefficient = data.GetPredCoefficient()

        nv = data.predVertices.shape[0]
        ne = data.predEdges.shape[0]
        nf = data.predFaces.shape[0]
        coefficient = 1 - coefficient*2
        coefficient[:nv + ne + nf] *= weight_unary
        coefficient[nv + ne + nf + nv * ne + nf * ne + nf * nv: ne + nv + nf + nv * ne + nf * ne + nf * nv + ne] *= weight_open
        # print(np.sum(data.predFE.reshape([-1,]) * coefficient[nv + ne + nf + ne * nv + nv * nf:nv + ne + nf + ne * nv + nv * nf + nf * ne]))

        vertex_valid_prob, edge_valid_prob, face_valid_prob, ev_connectivity, fv_connectivity, fe_connectivity, openness = gurobi_optimize(coefficient, nv, ne, nf)
        data_to_save = dict()
        data_to_save['vertices'] = data.predVertices
        data_to_save['vertex_valid_prob'] = vertex_valid_prob
        data_to_save['edge_valid_prob'] = edge_valid_prob
        data_to_save['face_valid_prob'] = face_valid_prob
        data_to_save['EV'] = ev_connectivity.reshape([ne, nv])
        data_to_save['FV'] = fv_connectivity.reshape([nf, nv])
        data_to_save['FE'] = fe_connectivity.reshape([nf, ne])
        data_to_save['Openness'] = openness

        with open(str(saving_dir / stem + "_extraction.npz"), "wb") as data_extraction:
            np.savez(data_extraction, **data_to_save)
    # print(np.sum(vertex_valid_prob * coefficient[:nv]))
    # print(np.sum(edge_valid_prob * coefficient[nv:nv + ne]))
    # print(np.sum(face_valid_prob * coefficient[nv + ne:nv + ne + nf]))
    # print(np.sum(ev_connectivity * coefficient[nv + ne + nf:nv + ne + nf + ne * nv]))
    # print(np.sum(fv_connectivity * coefficient[nv + ne + nf + ne * nv:nv + ne + nf + ne * nv + nv * nf]))
    # print(np.sum(fe_connectivity * coefficient[nv + ne + nf + ne * nv + nv * nf:nv + ne + nf + ne * nv + nv * nf + nf * ne]))
    # print(np.sum(openness * coefficient[nv + ne + nf + ne * nv + nv * nf + nf * ne:nv + ne + nf + ne * nv + nv * nf + nf * ne + ne]))
    
    # for i in range(nf):
    #     fe_connectivity[i * ne + 7] = 0.0
    # print(np.sum(fe_connectivity * coefficient[nv + ne + nf + ne * nv + nv * nf:nv + ne + nf + ne * nv + nv * nf + nf * ne]))

    
    # filepath = GetExpDataPath('./experiments/prediction_output/')
    # for file in filepath:
    #     data = OptData(file, optGeomWeight=0.5)
    #     if data.fileNumber == '00000093':
    #         pass
    #     coefficient = data.GetPredCoefficient()
        
    #     nv = data.predVertices.shape[0]
    #     ne = data.predEdges.shape[0]
    #     nf = data.predFaces.shape[0]
    #     coefficient = 1 - coefficient*2
    #     coefficient[:nv + ne + nf] *= weight_unary
    #     coefficient[nv + ne + nf + nv * ne + nf * ne + nf * nv: ne + nv + nf + nv * ne + nf * ne + nf * nv + ne] *= weight_open
    #     vertex_valid_prob, edge_valid_prob, face_valid_prob, ev_connectivity, fv_connectivity, fe_connectivity, openness = gurobi_optimize(coefficient, nv, ne, nf)
        
    #     data_to_save = dict()
    #     data_to_save['vertices'] = data.predVertices
    #     data_to_save['vertex_valid_prob'] = vertex_valid_prob
    #     data_to_save['edge_valid_prob'] = edge_valid_prob
    #     data_to_save['face_valid_prob'] = face_valid_prob
    #     data_to_save['EV'] = ev_connectivity.reshape([ne, nv])
    #     data_to_save['FV'] = fv_connectivity.reshape([nf, nv])
    #     data_to_save['FE'] = fe_connectivity.reshape([nf, ne])
    #     data_to_save['Openness'] = openness
        
    #     with open(os.path.join('./experiments/prediction_output/', f'{data.fileNumber}_optimized.npz'), "wb") as data_extraction:
    #         np.savez(data_extraction, **data_to_save)
    
    
    # for key in data.keys():
    #     print(key)
    
    # data = construct_data()
    
    # nv = 6
    # ne = 9
    # nf = 3
    
    # c = []
    # curve_corner_similarity_geom = get_curve_corner_similarity_geom(data)
    # patch_curve_similarity_geom = get_patch_curve_similarity_geom(data)
    # patch_corner_similarity_geom = get_patch_corner_similarity_geom(data)
    
    # c = np.concatenate([
    #     data['corners']['prediction']['valid_prob'].astype(np.float64),
    #     data['curves']['prediction']['valid_prob'].astype(np.float64),
    #     data['patches']['prediction']['valid_prob'].astype(np.float64),
    #     (0.5 * curve_corner_similarity_geom.astype(np.float64) + 0.5 * data['curve_corner_similarity'].astype(np.float64)).reshape([-1]),
    #     (0.5 * patch_corner_similarity_geom.astype(np.float64) + 0.5 * data['patch_corner_similarity'].astype(np.float64)).reshape([-1]),
    #     (0.5 * patch_curve_similarity_geom.astype(np.float64) + 0.5 * data['patch_curve_similarity'].astype(np.float64)).reshape([-1]),
    #     1.0 - data['curves']['prediction']['closed_prob'].astype(np.float64) * data['curves']['prediction']['valid_prob'].astype(np.float64)
    # ])
    
    # c = 1 - c*2
    # c[:nv + ne + nf] *= weight_unary
    # c[nv + ne + nf + nv * ne + nf * ne + nf * nv: ne + nv + nf + nv * ne + nf * ne + nf * nv + ne] *= weight_open
    
    # vertex_valid_prob, edge_valid_prob, face_valid_prob, ev_connectivity, fv_connectivity, fe_connectivity, openness = gurobi_optimize(c, nv, ne, nf)

# TODO: Persistence diagram crap then results and discuss read ellas paper

import numpy as np
import plotly.graph_objects as go

def plotting_struct(structure):
    x, y, z = structure[:,0] , structure[:,1], structure[:,2]
    fig = go.Figure(data=[go.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8
        )
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    fig.show()
    
def Structure(a_dist, b_dist, c_dist,
              a_atoms, b_atoms, c_atoms,
              gamma, alpha, beta,
              mid_ab_true, mid_ac_true, body_cetered, tolerance=1e-10
              ):
    
    vertices = []
    
    for a in range(a_atoms):
        for b in range(b_atoms):
            for c in range(c_atoms):
                vertices.append([(a * a_dist) + (np.cos(np.deg2rad(180 - gamma)) * b),
                                 (b * b_dist) + (np.cos(np.deg2rad(180 - alpha)) * c),
                                 (c * c_dist) + (np.cos(np.deg2rad(180 - beta)) * a)])  
                
    # Adds middle point if its on the side of ab (side of xy)
    if mid_ab_true == True:
        for a in range(a_atoms-1):
            for b in range(b_atoms-1):
                for c in range(c_atoms):
                    vertices.append([(a * a_dist) + a_dist/2 + (np.cos(np.deg2rad(180 - gamma)) * b),
                                     (b * b_dist) + b_dist/2 + (np.cos(np.deg2rad(180 - alpha)) * c),
                                     (c * c_dist)])
    
    # Adds middle point if its on the side of ac (side xz)
    if mid_ac_true == True:
        for a in range(a_atoms-1):
            for b in range(b_atoms):
                for c in range(c_atoms-1):
                    vertices.append([(a * a_dist) + a_dist/2 + (np.cos(np.deg2rad(180 - gamma)) * b),
                                     (b * b_dist),
                                     (c * c_dist) + c_dist/2 + (np.cos(np.deg2rad(180 - beta)) * a)])
                    
    if mid_ac_true == True & mid_ab_true == True:
        for a in range(a_atoms):
            for b in range(b_atoms -1):
                for c in range(c_atoms-1):
                    vertices.append([(a * a_dist),
                                     (b * b_dist) + b_dist/2 + (np.cos(np.deg2rad(180 - alpha)) * c),
                                     (c * c_dist) + c_dist/2 + (np.cos(np.deg2rad(180 - beta)) * a)])
    
    # Adds middle point in the center of the volume 
    if body_cetered == True:
        for a in range(a_atoms-1):
            for b in range(b_atoms-1):
                for c in range(c_atoms -1):
                    vertices.append([(a * a_dist) + a_dist/2,
                                     (b * b_dist) + b_dist/2,
                                     (c * c_dist) + c_dist/2])
                    
    vertices = np.array(vertices)
    vertices[np.abs(vertices) < tolerance] = 0
                            
    return vertices

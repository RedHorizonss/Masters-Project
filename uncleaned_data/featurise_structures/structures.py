# TODO: Possibly change this to a class that can be used to generate structures

import numpy as np
import plotly.graph_objects as go
import pandas as pd

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
    
def Structure(a_atoms, b_atoms, c_atoms,
              a_dist, b_dist, c_dist,
              alpha, beta, gamma,
              mid_ab_true, mid_ac_true, body_cetered, tolerance=1e-10
              ):
    
    vertices = []
    
    for a in range(a_atoms):
        for b in range(b_atoms):
            for c in range(c_atoms):
                vertices.append([(a * a_dist) + (np.cos(np.deg2rad(180 - alpha)) * b),
                                 (b * b_dist) + (np.cos(np.deg2rad(180 - beta)) * c),
                                 (c * c_dist) + (np.cos(np.deg2rad(180 - gamma)) * a)])  
                
    # Adds middle point if its on the side of ab (side of xy)
    if mid_ab_true == True:
        for a in range(a_atoms-1):
            for b in range(b_atoms-1):
                for c in range(c_atoms):
                    vertices.append([(a * a_dist) + a_dist/2 + (np.cos(np.deg2rad(180 - alpha)) * b),
                                     (b * b_dist) + b_dist/2 + (np.cos(np.deg2rad(180 - beta)) * c),
                                     (c * c_dist)])
    
    # Adds middle point if its on the side of ac (side xz)
    if mid_ac_true == True:
        for a in range(a_atoms-1):
            for b in range(b_atoms):
                for c in range(c_atoms-1):
                    vertices.append([(a * a_dist) + a_dist/2 + (np.cos(np.deg2rad(180 - alpha)) * b),
                                     (b * b_dist),
                                     (c * c_dist) + c_dist/2 + (np.cos(np.deg2rad(180 - gamma)) * a)])
                    
    if mid_ac_true == True & mid_ab_true == True:
        for a in range(a_atoms):
            for b in range(b_atoms -1):
                for c in range(c_atoms-1):
                    vertices.append([(a * a_dist),
                                     (b * b_dist) + b_dist/2 + (np.cos(np.deg2rad(180 - beta)) * c),
                                     (c * c_dist) + c_dist/2 + (np.cos(np.deg2rad(180 - gamma)) * a)])
    
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

def make_dataset(a, b, c, alpha, beta, gamma, name):
    data = {
        'a': a,
        'b': b,
        'c': c,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'structure type': name
        }
    
    df = pd.DataFrame(data)
    return df

def randomise_sides(length, num_of_diff_sides):
    sides_df = pd.DataFrame(columns=['a', 'b', 'c'])
    
    while len(sides_df) < length:
        side_a = round(np.random.uniform(3, 9.0), 3)
        if num_of_diff_sides == 1:
            side_b = side_a
            side_c  = round(np.random.uniform(3, 9.0), 3)
            
            if side_a != side_c:
                sides_df = sides_df.append({'a': side_a, 'b': side_b, 'c': side_c}, ignore_index=True)
            
        elif num_of_diff_sides == 2:
            side_b = round(np.random.uniform(3, 9.0), 3)
            side_c  = round(np.random.uniform(3, 9.0), 3)
            
            if side_a != side_b != side_c:
                sides_df = sides_df.append({'a': side_a, 'b': side_b, 'c': side_c}, ignore_index=True)
                
        else:
            side_b = side_a
            side_c = side_a
            sides_df = sides_df.append({'a': side_a, 'b': side_b, 'c': side_c}, ignore_index=True)
            
    return sides_df

def randomise_angles(length, randomise_1, randomise_3, same_3_randomised):
    if randomise_1 == True:
        angles = []
        while len(angles) < length:
            ang_a = np.random.uniform(45, 80)
            ang_a = round(ang_a, 1)
            angles.append(ang_a)
            
    elif randomise_3 == True:
        angles = []
        while len(angles) < length:
            ang_a = np.random.uniform(45, 80)
            ang_a = round(ang_a, 1)
            ang_b = np.random.uniform(45, 80)
            ang_b = round(ang_b, 1)
            ang_y = np.random.uniform(45, 80)
            ang_y = round(ang_y, 1)
            
            if ang_a != ang_b != ang_y:
                angles.append([ang_a, ang_b, ang_y])
                
    elif same_3_randomised == True:
        angles = []
        while len(angles) < length:
            ang_a = np.random.uniform(45, 80)
            ang_a = round(ang_a, 1)
            ang_b = ang_a
            ang_y = ang_a
            
            angles.append([ang_a, ang_b, ang_y])
            
    else:
        print('Please choose a valid option')
        
    return angles

def finalise_dataset(length):

    sides = randomise_sides(length, 0)
    cubic = make_dataset(sides["a"], sides["b"], sides["c"], 90, 90, 90, "cubic")

    sides = randomise_sides(length, 0)
    hexagonal = make_dataset(sides["a"], sides["b"], sides["c"], 90, 90, 120, "hexagonal")

    sides = randomise_sides(length, 1)
    tetragonal = make_dataset(sides["a"], sides["b"], sides["c"], 90, 90, 90, "tetragonal")

    sides = randomise_sides(length, 2)
    orthorhombic = make_dataset(sides["a"], sides["b"], sides["c"], 90, 90, 90, "orthorhombic")

    sides = randomise_sides(length, 0)
    angles = randomise_angles(length, False, False, True)
    angles = pd.DataFrame(angles, columns=['alpha', 'beta', 'gamma'])
    rhomobohedral = make_dataset(sides["a"], sides["b"], sides["c"], angles["alpha"], 
                                angles["beta"], angles["gamma"], "rhomobohedral")

    sides = randomise_sides(length, 2)
    angles = randomise_angles(length, True, False, False)
    monoclinc = make_dataset(sides["a"], sides["b"], sides["c"], 90, angles, 90, "monoclinc")

    sides = randomise_sides(length, 2)
    angles = randomise_angles(length, False, True, False)
    angles = pd.DataFrame(angles, columns=['alpha', 'beta', 'gamma'])

    triclinic = make_dataset(sides["a"], sides["b"], sides["c"],
                                angles["alpha"], angles["beta"], angles["gamma"], "triclinic")


    final_df = pd.concat([cubic, hexagonal, rhomobohedral, tetragonal, orthorhombic, monoclinc, triclinic]
                         ,ignore_index=True)
    
    return final_df
    
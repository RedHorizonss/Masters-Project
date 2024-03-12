import os
import numpy as np
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_union, Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import Amplitude


class BravaisLattice():
    def __init__(self, a_dist, b_dist, c_dist, alpha, beta, gamma, mid_ab_true = False, mid_ac_true = False, body_centered = False, name = "lattice"):
        self.a_dist = a_dist
        self.b_dist = b_dist
        self.c_dist = c_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.body_centered = body_centered
        self.mid_ab = mid_ab_true
        self.mid_ac = mid_ac_true
        self.all_faces = mid_ab_true and mid_ac_true
        
        self.unit_cell = self.get_coords(2)
        self.name = name
        
    def get_unit_cell(self):
        return self.unit_cell
    
    def get_coords(self, num_atoms):
        tolerance = 1e-10
        
        if isinstance(num_atoms, list):
            a_atoms, b_atoms, c_atoms = num_atoms
        else:
            a_atoms = b_atoms = c_atoms = int(num_atoms)
        
        vertices = []
        for a in range(a_atoms):
            for b in range(b_atoms):
                for c in range(c_atoms):
                    vertices.append([(a * self.a_dist) + (np.cos(np.deg2rad(180 - self.alpha)) * b),
                                    (b * self.b_dist) + (np.cos(np.deg2rad(180 - self.beta)) * c),
                                    (c * self.c_dist) + (np.cos(np.deg2rad(180 - self.gamma)) * a)])
        
            # Adds middle point if its on the side of ab (side of xy)
        if self.mid_ab == True:
            for a in range(a_atoms-1):
                for b in range(b_atoms-1):
                    for c in range(c_atoms):
                        vertices.append([(a * self.a_dist) + self.a_dist/2 + (np.cos(np.deg2rad(180 - self.alpha)) * b),
                                        (b * self.b_dist) + self.b_dist/2 + (np.cos(np.deg2rad(180 - self.beta)) * c),
                                        (c * self.c_dist)])
        
        # Adds middle point if its on the side of ac (side xz)
        if self.mid_ac == True:
            for a in range(a_atoms-1):
                for b in range(b_atoms):
                    for c in range(c_atoms-1):
                        vertices.append([(a * self.a_dist) + self.a_dist/2 + (np.cos(np.deg2rad(180 - self.alpha)) * b),
                                        (b * self.b_dist),
                                        (c * self.c_dist) + self.c_dist/2 + (np.cos(np.deg2rad(180 - self.gamma)) * a)])
                        
        if self.all_faces == True:
            for a in range(a_atoms):
                for b in range(b_atoms -1):
                    for c in range(c_atoms-1):
                        vertices.append([(a * self.a_dist),
                                        (b * self.b_dist) + self.b_dist/2 + (np.cos(np.deg2rad(180 - self.beta)) * c),
                                        (c * self.c_dist) + self.c_dist/2 + (np.cos(np.deg2rad(180 - self.gamma)) * a)])
        
        # Adds middle point in the center of the volume 
        if self.body_centered == True:
            for a in range(a_atoms-1):
                for b in range(b_atoms-1):
                    for c in range(c_atoms -1):
                        vertices.append([(a * self.a_dist) + self.a_dist/2,
                                        (b * self.b_dist) + self.b_dist/2,
                                        (c * self.c_dist) + self.c_dist/2])
        
        vertices = np.array(vertices)
        vertices[np.abs(vertices) < tolerance] = 0
        
        return vertices
    
    def plotting_struct(self, coords = None):
        if coords is None:
            coords = self.unit_cell
            
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
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
    
    def add_edge_trace(self, fig, edges, line_color='grey', line_width=10, dash='solid'):
        for edge in edges:
            x_line = [self.unit_cell[edge[0], 0], self.unit_cell[edge[1], 0]]
            y_line = [self.unit_cell[edge[0], 1], self.unit_cell[edge[1], 1]]
            z_line = [self.unit_cell[edge[0], 2], self.unit_cell[edge[1], 2]]

            fig.add_trace(go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='lines',
                line=dict(color=line_color, width=line_width, dash=dash),
                showlegend=False
            ))

    def draw_fancy_unitcell(self, show = True, save_image = False, fig_height = 1200, fig_width = 1500):
        if self.gamma == 120 and self.alpha == 90 and self.beta == 90:
            # need to switch things to make it look pretty
            self.gamma = 90
            self.alpha = 120
            self.unit_cell = self.get_coords([3, 3, 2])[2:-2]
        
        fig = go.Figure()

        # Plot the vertices
        fig.add_trace(go.Scatter3d(
            x=self.unit_cell[:, 0],
            y=self.unit_cell[:, 1],
            z=self.unit_cell[:, 2],
            mode='markers',
            marker=dict(color='black', size=12, opacity=0.8),
            showlegend=False
        ))
        
        if self.gamma == 90 and self.alpha == 120 and self.beta == 90:
            dashed_edges = [
            [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
            [4, 5], [5, 7], [7, 6], [6, 4],  # Side faces
            [0, 4], [1, 5], [2, 6], [3, 7],  # Top edges
            [12,13], [13,11], [11,10], [10,12],
            [10,4], [11,5], [12,6], [13,7]]
        
            edges = [[8,2], [9,3], [2,3],[8,9],
                    [9,13], [13,12], [12,8], [2, 6],
                    [3, 7],[6,7], [7,13], [6,12]]
            
            self.add_edge_trace(fig, edges)
            self.add_edge_trace(fig, dashed_edges, dash='dash')
            # Im putting the hexagonal cell back to normal here
            self.alpha = 90
            self.gamma = 120
            self.unit_cell = self.get_coords(2)
        else:
            edges = [
                [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
                [4, 5], [5, 7], [7, 6], [6, 4],  # Side faces
                [0, 4], [1, 5], [2, 6], [3, 7],  # Top edges
            ]
            
            self.add_edge_trace(fig, edges)

            mid_ab_edge = [[0, 6], [1, 7], [2, 4], [3, 5]]
            mid_ac_edge = [[0, 5], [1, 4], [2, 7], [3, 6]]
            mid_bc_edge = [[0, 3], [1, 2], [4, 7], [5, 6]]

            if self.all_faces:
                self.add_edge_trace(fig, mid_ab_edge + mid_ac_edge + mid_bc_edge, line_width=5, dash='dash')
            elif self.mid_ab:
                self.add_edge_trace(fig, mid_ab_edge, line_width=5, dash='dash')
            elif self.mid_ac:
                self.add_edge_trace(fig, mid_ac_edge, line_width=5, dash='dash')

            if self.body_centered:
                center = np.mean(self.unit_cell, axis=0)
                body_edges = [[i, center] for i in range(self.unit_cell.shape[0])]
                # Plot the body edges
                for edge in body_edges:
                    x_line = [self.unit_cell[edge[0], 0], center[0]]
                    y_line = [self.unit_cell[edge[0], 1], center[1]]
                    z_line = [self.unit_cell[edge[0], 2], center[2]]

                    fig.add_trace(go.Scatter3d(
                        x=x_line,
                        y=y_line,
                        z=z_line,
                        mode='lines',
                        line=dict(color='grey', width=5, dash='dash'),
                        showlegend=False
                    ))

        # Customize layout
        fig.update_layout(
            {'paper_bgcolor': "rgba(0,0,0,0)"},
            scene=dict(
                xaxis=dict(title='', title_font=dict(size=20), ticks='', showticklabels=False),
                yaxis=dict(title='', title_font=dict(size=20), ticks='', showticklabels=False),
                zaxis=dict(title='', title_font=dict(size=20), ticks='', showticklabels=False),
                camera=dict(
                    eye=dict(x=1.5, y=2, z=1.5)
                ),
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=fig_height,
            width=fig_width,
            template='plotly_white'
        )
        if show:
            fig.show()
        if save_image:
            if not os.path.exists("lattice_images"):
                os.mkdir("lattice_images")
            fig.write_image(f"lattice_images/{self.name}.png")

class PresistentHomologyFeatures():
    def __init__(self, coords, name = "structure"):
        if isinstance(coords, list):
            self.list_of_coords = coords
        if isinstance(coords, np.ndarray):
            self.coords = coords
            self.diagrams_basic = self.compute_persistence_diagrams()
        self.name = name
        
    def compute_persistence_diagrams(self, coords = None, n_jobs = 1):
        # Track H0, H1, H2: connections, 1d voids and 2d voids
        homology_dimensions = [0, 1, 2]

        # Collapse edges to speed up H2 persistence calculation
        persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=homology_dimensions,
            n_jobs= n_jobs,
            collapse_edges=True,
        )
        
        # If no coords are given, use the ones from the class
        if coords is not None:
            reshaped_coords=coords[None, :, :]
        else:
            reshaped_coords=self.coords[None, :, :]
        
        diagrams_basic = persistence.fit_transform(reshaped_coords)
        return diagrams_basic
    
    def make_pipeline(self):

        metrics = [
            {"metric": metric}
            for metric in ["bottleneck", "wasserstein", "landscape", "persistence_image"]
        ]

        # Concatenate to generate 3 + 3 + (4 x 3) = 18 topological features
        feature_union = make_union(
            PersistenceEntropy(normalize=True),
            NumberOfPoints(n_jobs=1),
            *[Amplitude(**metric, n_jobs=1) for metric in metrics]
        )

        ## then we use a pipeline to transform, the data and spit i out
        # mwah hahahahaha
        pipe = Pipeline(
            [
                ("features", feature_union)
            ]
        )
            
        return pipe

    def featurising_coords(self):
        topol_feat_list = []
        pipe = self.make_pipeline()

        for coordinates in self.list_of_coords:
            diagrams_basic = self.compute_persistence_diagrams(coordinates)
            X_basic = pipe.fit_transform(diagrams_basic)
            # topology feat list stores the topological features for each structure
            topol_feat_list.append([x for x in X_basic[0]])
        
        # topol feat mat is a matrix of topological features
        topol_feat_mat = np.array(topol_feat_list)
        
        return topol_feat_mat, topol_feat_list
    
    def diagram_manipulation(self):
        diagram = self.diagrams_basic[0]
        
        birth_death_pairs = diagram[diagram[:, 0] != diagram[:, 1]]
        no_homology_pairs = birth_death_pairs[:,:2]
        
        posinfinite_mask = np.isposinf(no_homology_pairs)
        neginfinite_mask = np.isneginf(no_homology_pairs)
        
        max_val = np.max(np.where(posinfinite_mask, -np.inf, no_homology_pairs))
        min_val = np.min(np.where(neginfinite_mask, np.inf, no_homology_pairs))
        
        return diagram, birth_death_pairs, max_val, min_val
        
    def plot_presistent_diagrams(self, 
                                 show = True,
                                 save_image = False,
                                 shift_annotation_x = [[25, 25, 25],[25,25,25],[25]], 
                                 shift_annotation_y = [[17, 17,17],[17, 17,17],[17]]):
        
        diagram, birth_death_pairs, max_val, min_val = self.diagram_manipulation()
        
        homology_dimensions = np.unique(diagram[:,2])
        
        extra_space = 0.02 * (max_val - min_val)
        min_val_display = min_val - extra_space
        max_val_display = max_val + extra_space+0.1

        dotted_line = [[min_val_display, max_val_display], [min_val_display, max_val_display]]

        fig = go.Figure()

        # Add the scatter plot for each dimension
        for index, dim in enumerate(homology_dimensions):
            subdiagram = birth_death_pairs[birth_death_pairs[:,2] == dim]
            unique, counts = np.unique(subdiagram, axis=0, return_counts=True)
            
            if unique.shape[0] == 0:
                continue
            
            x = unique[:,0]
            y = unique[:,1]
            
            col = ['rgb(53, 183, 121)' if int(x) == 0 else 'rgb(49, 104, 142)' if int(x) == 1 else 'rgb(68, 1, 84)' for x in unique[:,2]]
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    color=col,
                    size=12
                ),
                name=r'$\large{{H_{}}}$'.format(int(dim))
                
            ))
            
            shift_ann_x = shift_annotation_x[index]
            shift_ann_y = shift_annotation_y[index]
            
            for i in range(len(counts)):
                annotation = f"m = {counts[i]}"
                fig.add_annotation(
                    x=x[i],
                    y=y[i], 
                    text=annotation,
                    showarrow=False,
                    arrowhead=1,
                    xshift=shift_ann_x[i],
                    yshift=shift_ann_y[i],
                     font=dict(
                         size=18,
                         color="black"
                        )
                )
                
        # Add the dotted line
        fig.add_trace(go.Scatter(
            x=dotted_line[0],
            y=dotted_line[1],
            mode='lines',
            line=dict(
                dash='dash',
                color='grey'
            ),
            name = "persistence"
        ))

        # Set the layout
        fig.update_layout(
            xaxis=dict(
                title='Birth',
                title_font=dict(size=20),
                tickfont=dict(size=18), 
                tickmode = 'linear',
                dtick=round(max_val - min_val) / 5, 
                tickformat=".1f",
                ticks="outside",
                ticklen=5,
                tickwidth=2,
            ),
            yaxis=dict(
                title='Death',
                title_font=dict(size=20),
                tickfont=dict(size=18), 
                tickmode = 'linear',
                dtick=round(max_val - min_val) / 5, 
                tickformat=".1f",
                ticks="outside",
                ticklen=5,
                tickwidth=2,
            ),
            font=dict(
                family="Helvetica",
                size=18,
                color="black",
            ),
            margin=dict(l=10, r=10, b=10, t=10),
            showlegend=True,
            width=500,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=go.layout.Legend(
                itemsizing='constant',
                x = 0.6,
                y = 0.2,
                traceorder="normal",
                bgcolor='rgba(0,0,0,0)',
                font=dict(size= 20)
            )
        )
        
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', range=[min_val_display, max_val_display])
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', range=[min_val_display, max_val_display])
                
        if show:
            fig.show()
        if save_image:
            if not os.path.exists("PHF_images"):
                os.mkdir("PHF_images")
            fig.write_image(f"PHF_images/{self.name}.png")
        
    def plot_barcode_plots(self, show = True, save_image = False,):
        _, birth_death_pairs, max_val, min_val = self.diagram_manipulation()

        x_left = birth_death_pairs[:, 0]
        x_right = birth_death_pairs[:, 1]
        
        homology_dimensions = birth_death_pairs[:, 2]
        
        # Setting up colors for different homology dimensions
        colors = {0: 'rgb(53, 183, 121)', 1: 'rgb(49, 104, 142)', 2: 'rgb(68, 1, 84)'}

        # Creating traces for each homology dimension
        traces = []
        legend_items = set()  # Initialize set to store unique legend items
        for i, (left, right, dim) in enumerate(zip(x_left, x_right, homology_dimensions)):
            if dim not in legend_items:  # Check if the legend item is already added
                traces.append(go.Scatter(
                    x=[left, right],
                    y=[-i, -i],
                    mode='lines',
                    line=dict(color=colors[dim], width=2),
                    name=r'$\Large{{\beta_{}}}$'.format(int(dim))
                ))
                legend_items.add(dim)  # Add the legend item to the set
            else:  # Add the line without a legend item
                traces.append(go.Scatter(
                    x=[left, right],
                    y=[-i, -i],
                    mode='lines',
                    line=dict(color=colors[dim], width=2),
                    showlegend=False
                ))

        # Setting layout options
        layout = go.Layout(
            xaxis=dict(
                title='Filter / Å',
                title_font=dict(size=20),
                tickfont=dict(size=18), 
                tickmode='linear',
                dtick=round(max_val - min_val) / 5,  # Set tick intervals based on data range
                tickformat=".1f",
                ticks="inside",
                ticklen=5,
                tickwidth=2,
            ),
            yaxis=dict(
                title='Multiplicity',
                showticklabels=False,
            ),
            font=dict(
                family="Helvetica",
                size=18,
                color="black",
            ),
            margin=dict(l=10, r=10, b=10, t=10),
            width=500,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=go.layout.Legend(
                itemsizing='constant',
                x = 1.02,
                y=1,
                font=dict(size= 20)
            )
        )
        

        # Creating figure
        fig = go.Figure(data=traces, layout=layout)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', range=[min_val, max_val], title_font_family="Helvetica")
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

        if show:
            fig.show()
        if save_image:
            if not os.path.exists("barcode_images"):
                os.mkdir("barcode_images")
            fig.write_image(f"barcode_images/{self.name}.png")
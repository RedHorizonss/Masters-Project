import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_union, Pipeline

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import Amplitude

class BravaisLattice():
    """Class to create a Bravais Lattices"""
    def __init__(self, a_dist, b_dist, c_dist, alpha, beta, gamma, mid_ab_true = False, mid_ac_true = False, body_centered = False, name = "lattice"):
        """Initialises the Bravais Lattice

        Args:
            a_dist (float): distance between atoms in the a direction (x axis)
            b_dist (float): distance between atoms in the b direction (y axis)
            c_dist (float): distance between atoms in the c direction (z axis)
            alpha (float): angle between b and c
            beta (float): angle between a and c
            gamma (float): angle between a and b
            mid_ab_true (boolean, optional): If a lattice point is found on face with lengths a and b. Defaults to False.
            mid_ac_true (boolean, optional): If a lattice point is found on face with lengths a and c. Defaults to False.
            body_centered (boolean, optional): If a lattice point is found in the body of the lattice. Defaults to False.
            name (string, optional): Name of files to be saved as. Defaults to "lattice".
        """
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
        """runs the get_coords function to get the unit cell

        Returns:
            np.array: unit cell coordinates
        """
        return self.unit_cell
    
    def get_coords(self, num_atoms):
        """provides the cooordinates of the lattice

        Args:
            num_atoms (int or list of ints): number of atoms in the lattice. 
            If a list is given, the first element is the number of atoms in the a direction, the second in the b direction and the third in the c direction

        Returns:
            np.array: coordinates of the lattice
        """
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
        """plots the structure of the lattice

        Args:
            coords (np.array, optional): the cordinates of alttice. Defaults to None.
        """
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
        
        # Axis names
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            )
        )
        fig.show()

    def add_edge_trace(self, fig, edges, line_color='grey', line_width=10, dash='solid'):
        """adds the edges to the plot

        Args:
            fig (plotly figure): the figure to add the edges to
            edges (list): the edges to add to the plot
            line_color (string, optional): color of the edges. Defaults to 'grey'.
            line_width (float, optional): width of the line. Defaults to 10.
            dash (string, optional): looks for the edge lines. Defaults to 'solid'.
        """
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
        """draw the fancy unit cell, with edges and vertices

        Args:
            show (boolean, optional): to shw the figrue or not. Defaults to True.
            save_image (boolean, optional): to save the image or not. Defaults to False.
            fig_height (int, optional): hieght of figure. Defaults to 1200.
            fig_width (int, optional): width of figure. Defaults to 1500.
        """
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
            
            # Adds the mid points
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
    """Class to create the persistent homology features of a structure
    Functions, compute_persistence_diagrams, make_pipeline, featurising_coords were adapted from Ella Gales G and T repo.
    Functions diagram_manipulation, plot_presistent_diagrams, plot_barcode_plots are adapted from Giotto-TDA.
    """
    def __init__(self, coords, name = "structure"):
        """Initialises the class

        Args:
            coords (list): list of coordinates of the structure
            name (string, optional): name to save files as. Defaults to "structure".
        """
        if isinstance(coords, list):
            self.list_of_coords = coords
        if isinstance(coords, np.ndarray):
            self.coords = coords
            self.diagrams_basic = self.compute_persistence_diagrams()
        self.name = name
        
    def compute_persistence_diagrams(self, coords = None, n_jobs = 1):
        """computes the persistence diagrams of the structure

        Args:
            coords (list, optional): coordinates of structure. Defaults to None.
            n_jobs (int, optional): the number of CPUs to run the code on. Defaults to 1.

        Returns:
            np.array: persistence diagrams
        """
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
        """makes the pipeline for the topological features

        Returns:
            pipe: pipeline for the topological features
        """
        # Define the metrics to be used for the topological features
        metrics = [
            {"metric": metric}
            for metric in ["bottleneck", "wasserstein", "landscape", "persistence_image"]
        ]

        # 3 features made for each metric, 1 for each homology dimension
        feature_union = make_union(
            PersistenceEntropy(normalize=True),
            NumberOfPoints(n_jobs=1),
            *[Amplitude(**metric, n_jobs=1) for metric in metrics]
        )
        
        # Create the pipeline
        pipe = Pipeline(
            [
                ("features", feature_union)
            ]
        )
            
        return pipe

    def featurising_coords(self):
        """featurises the coordinates of the structure

        Returns:
            np.array and list: matrix of topological features and list of topological features
        """
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
        """manipulates the diagram to get the birth-death pairs

        Returns:
            diagram manipulated: the diagram with the birth-death pairs and max and min values
        """
        # Extracts the first diagram
        diagram = self.diagrams_basic[0]
        
        # Extracts the birth-death pairs
        birth_death_pairs = diagram[diagram[:, 0] != diagram[:, 1]]
        no_homology_pairs = birth_death_pairs[:,:2]
        
        #removes any infinite values
        posinfinite_mask = np.isposinf(no_homology_pairs)
        neginfinite_mask = np.isneginf(no_homology_pairs)
        
        # Replaces infinite values with max and min values
        max_val = np.max(np.where(posinfinite_mask, -np.inf, no_homology_pairs))
        min_val = np.min(np.where(neginfinite_mask, np.inf, no_homology_pairs))
        
        return diagram, birth_death_pairs, max_val, min_val
        
    def plot_presistent_diagrams(self, show = True, save_image = False,shift_annotation_x = [[25, 25, 25],[25,25,25],[25]], shift_annotation_y = [[17, 17,17],[17, 17,17],[17]]):
        """plots the persistent diagrams

        Args:
            show (bool, optional): shows the figure. Defaults to True.
            save_image (bool, optional): saves the image if true. Defaults to False.
            shift_annotation_x (list, optional): lists to shift each annoted feature in the x axis. Defaults to [[25, 25, 25],[25,25,25],[25]].
            shift_annotation_y (list, optional): lists to shift the annotation for each feature in the y axis. Defaults to [[17, 17,17],[17, 17,17],[17]].
        """
        
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
            fig.write_image(f"PHF_images/{self.name}.png", width=500, height=500, scale=3)
        
    def plot_barcode_plots(self, show = True, save_image = False,):
        """plots the barcode plots

        Args:
            show (bool, optional): shows the figure. Defaults to True.
            save_image (bool, optional): save the image if true. Defaults to False.
        """
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
            fig.write_image(f"barcode_images/{self.name}.png", width=500, height=500, scale=3)
            
class randomforests():
    def __init__(self, df, features, target, test_size = 0.2, random_state = 42, name = "model", stratify = False):
        """Initialises the class

        Args:
            df (_type_): dataframe
            features (list): list of string of the column names to be uysed as the features
            target (string): name of column to be target
            test_size (float, optional): size of test set to be used. Defaults to 0.2.
            random_state (int, optional): the random state to be used in the random forests, if set to none will use none. Defaults to 42.
            name (string, optional): name of plots to be saved as. Defaults to "model".
            stratify (bool, optional): for classification models set to stratify to enable the model to evenly pick out each class for thr test set. Defaults to False.
        """
        self.df = df
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.name = name
        self.stratify = stratify
        
        # creates test abd training set from data
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        
    def split_data(self):
        """splits the data into test and training sets

        Returns:
            test and traing sets : X_train, X_test, y_train, y_test
        """
        X = self.df[self.features]
        y = self.df[self.target]
        
        # if stratify is set to true, the model will evenly pick out each class for the test set ONLY FOR CLASSIFICATION MODELS
        if self.stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=self.test_size, random_state=self.random_state,
                                                                stratify= y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=self.test_size, random_state=self.random_state)
        
        return X_train, X_test, y_train, y_test
    
    def train_classifier_model(self, n_estimators = 100,  max_depth=5, min_samples_leaf=1, min_samples_split=2):
        """trains a classifier model

        Args:
            n_estimators (int, optional): number of trees. Defaults to 100.
            max_depth (int, optional): maximum deapth of the trees. Defaults to 5.
            min_samples_leaf (int, optional): minimum data points on each leaf. Defaults to 1.
            min_samples_split (int, optional): minimum data points on the leaf to split. Defaults to 2.

        Returns:
            rf: the fitted model
        """
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth, 
                                    min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,
                                    random_state=self.random_state)
        
        rf.fit(self.X_train, self.y_train)
        
        return rf
    
    def train_regressor_model(self, n_estimators = 100,  max_depth=5, min_samples_leaf=1, min_samples_split=2):
        """trains a regressor model

        Args:
            n_estimators (int, optional): number of trees. Defaults to 100.
            max_depth (int, optional): maximum deapth of the trees. Defaults to 5.
            min_samples_leaf (int, optional): minimum data points on each leaf. Defaults to 1.
            min_samples_split (int, optional): minimum data points on the leaf to split. Defaults to 2.

        Returns:
            rf: the fitted model
        """
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth, 
                                    min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,
                                    random_state=self.random_state)
        
        rf.fit(self.X_train, self.y_train)
        
        return rf

    def train_regressor_model_grid_search(self, cv = 10, scoring = 'neg_mean_squared_error', 
                                          param_grid = {
                                              'n_estimators': [50, 100, 150],
                                              'max_depth': [None, 10, 20],
                                              'min_samples_split': [2, 5, 10],
                                              'min_samples_leaf': [1, 2, 4],
                                              'max_features': ['log2', 'sqrt']}):
        
        """trains a regressor model using grid search
        
        Args:
            cv (int, optional): number of cross validations. Defaults to 10.
            scoring (string, optional): scoring method. Defaults to 'neg_mean_squared_error'.
            param_grid (dict, optional): paramaters to be used in the grid search.

        Returns:
            model and paramaters: the best model and the best paramaters
        """
        
        rf = RandomForestRegressor(random_state=self.random_state)
        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(self.X_train, self.y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_regressor_model(self, model):
        """`evaluates the regressor model

        Args:
            model (model): the model to be evaluated

        Returns:
            evaluation metrics: mean absolute error, mean squared error, r2 score
        """
        
        y_pred = model.predict(self.X_test)
        
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        return mae, mse, r2
    
    def evaluate_classifier_model(self, model):
        """evaluates the classifier model

        Args:
            model (model): the model to be evaluated

        Returns:
            evaluation metrics: accuracy, f1, precision, recall, confusion matrix
        """
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        return accuracy, f1, precision, recall, conf_matrix
    
    def calc_cross_val_score(self, model, cv = 5, scoring = 'accuracy'):
        """calculates the cross validation score

        Args:
            model (_type_): the model to be used
            cv (int, optional): number of folds to implement. Defaults to 5.
            scoring (str, optional): scoring value. Defaults to 'accuracy'.

        Returns:
            scores: a list of cross validation scores
        """
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring)
        
        return scores
    
    def plot_confusion_matrix(self, target_names, conf_matrix, show = True, save_image = False, width = 800, height = 600):
        """plots the confusion matrix

        Args:
            target_names (list): names of target values 
            conf_matrix (matrix): the confusion matrix 
            show (bool, optional): show the plot Defaults to True.
            save_image (bool, optional): save the plot as a png. Defaults to False.
            width (int, optional): width of plot. Defaults to 800.
            height (int, optional): height of plot. Defaults to 600.
        """
        heatmap = go.Heatmap(z=conf_matrix,
                             x=target_names,
                             y=target_names,
                             colorscale='Viridis',
                             text=conf_matrix,
                             texttemplate="%{text}",
                             textfont={"size":22})

        layout = go.Layout(
            xaxis=dict(title='Predicted Class'),
            yaxis=dict(title='True Class'),
            width=width,
            height=height,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Helvetica', size=24, color='black'),
            margin=dict(l=5, r=5, b=5, t=10))

        # Create the figure
        fig = go.Figure(data=[heatmap], layout=layout)
        
        if show:
            fig.show()
        if save_image:
            if not os.path.exists("plots"):
                os.mkdir("plots")
            fig.write_image(f"plots/conf_matrix_{self.name}.png", width=width, height=height, scale=3)
            
    def plot_feature_importance(self, importances ,show = True, save_image = False, width = 800, height = 600):
        """plots the feature importance

        Args:
            importances (_type_): the feature importances
            show (bool, optional): to show the figure or not. Defaults to True.
            save_image (bool, optional): to save th image or not. Defaults to False.
            width (int, optional): width of figure. Defaults to 800.
            height (int, optional): height of figure. Defaults to 600.
        """
        
        # Get the feature names
        feature_names = [name.replace('_', ' ') for name in self.features]

        # Sort the feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Create the bar plot
        fig = go.Figure(data=go.Bar(
            x=[feature_names[i] for i in indices],
            y=importances[indices],
            marker_color='rgb(33, 145, 140)', 
            text= [f'{x:.2f}' for x in importances[indices]],
            textposition='auto',
        ))

        # Set the layout
        fig.update_layout(
            xaxis=dict(
                title="Features",
                showline=True,
                linewidth=5,
                linecolor='black',
                ticks='outside',
                tickson = "boundaries",
                tickwidth=3,
                ticklen=5
            ),
            yaxis=dict(
                title="Feature Importance",
                showline=True,
                linewidth=5,
                linecolor='black',
                ticks='inside',
                tickwidth=3,
                ticklen=5
            ),
            barmode='group',
            width=width,
            height=height,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Helvetica', size=24, color='black'),
            margin=dict(l=10, r=10, b=10, t=10),
            showlegend=False,
            )
        
        if show:
            fig.show()
        if save_image:
            if not os.path.exists("plots"):
                os.mkdir("plots")
            fig.write_image(f"plots/feature_importance_{self.name}.png", width=width, height=height, scale=3)
            
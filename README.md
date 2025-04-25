# Predicting Critical Temperature with Persistence Homology Features

## Description
The project predicts the critical temperature of superconducting materials by producing a random forest regression model. 
The main feature implemented was the crystal structure of these materials which were featurised using Persistent Homology.

## Installation
The enviroment used is stored within the requirements.yml file. All folders apart from presentation plots can be ran with this enviroment.

I would reccomend using a conda enviroment to set it up. Issues surrounding ipykernal can arise from this enviroment, please make sure to update these packages if so.

matplotlib was not used due to DLL errors that could not be resollved at the time.

A venv enviroment needs to be used to run presentation plots with enviroments : pandas, matplotlib, seaborn 

## Usage
The majority of code is within PHF_RF.py, with examples to run the code in each folder. To create bravais lattices see FinalMastersCode\bravais_lattice_experiment, both ipynb files present exmaples. 
To produce PH features and implement a random forest regression and classification model, see FinalMastersCode\model_supercon_experiments.

Ex4 files seperate creating the coordinates, featurising the coordinates, modelling the PH feature with random forests. 

- File Ex4_3DSC_coords.ipynb creates the coordinates, this is the first step.
- File Ex4_3DSC_featurise.ipynb featurises the coordinates, transforming the coordinates into PH features. Warning this code takes 5 hours to run on the full 3DSC_MP dataset.
- File Ex4_3DSC_models.ipynb, presents the final step where the PH features were implemented into a random forest regressor.

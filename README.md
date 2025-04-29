# seminar_project
This my project repo for CSE 5469

# Data 
Training/testing data unfortunetly could not be provided due to github file upload limits (github limts 1000 files per folder, and max file size of 25MB).

# Viewing Results
Since there is no data provided, you can view my results in either `eval_script.ipynb`, or the 'runs/detect' folder, where the results of the model evaluations are stored.

# Data Modification
The data modification folder contains all the scripts I made to diginatlly modify the dataset for the experiments.

# Training Scripts
The training scripts folder contains the scripts I used to train the YOLO models

# Models
The models folder contains the .pt files for the baseline and adversarially trained models.

# Runs
The runs folder contains the results of all the model evalutaions I ran for the project.
It contains sample images, F1 curves, confusion matrices, and more, for all evaluations.

# Eval_Script
If you download the data yourself in YOLO format, you can run `eval_script.ipynb` to see the results yourself. Just be sure that you are on the newest version of Ultralytics, as YOLO11 is a new architecture and requires the most up-to-date version, and also be sure to properly replace the file path with ones accurate to your machine.

You shouldn't have to train a new model, as I have provided `best.pt` and `mud_best.pt` which are the pytorch files for the original model and adversarially trained model, respectivly.


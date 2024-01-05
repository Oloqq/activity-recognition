# Activity recognition

The project's goal is to compare performance of various machine learning techniques in recognizing activity performed by a human based on IMU readings on their phone.

# How to run

## Setup
Project was developed with python 3.11.0

To get started, create a virtual environment,
```
python -m venv venv
```
and activate it with the script matching you operating system.

Install required packages
```
pip install -r requirements.txt
```

## Usage
You can interact with the data and models using jupyter notebooks

- Data visualization
  - [Visualization 1](./visualization1.ipynb)
  - [Visualization 2](./visualization2.ipynb)
- Data preparation
  - [Resampling](./resample.ipynb)
  - [Data trimming](./setting_trims.ipynb)
- Models
  - Logistic Regression, KNeighbors, SVC, DecisionTree, RandomForest
    - [using SLERP for quaterions](./models_ml_slerp.ipynb)
    - [using ordinary interpolation](./models_ml_lerp.ipynb)
  - [Long short-term memory (LSTM)](./models_LSTM.ipynb)

Additionaly, [time_per_activity.py](./time_per_activity.py) helps with summarizing total time for each activity type. [grid.py](./grid.py) enables the search of hiperparameter space, and [hiperparam_summary.py](./hiperparam_summary.py) gathers it's results.
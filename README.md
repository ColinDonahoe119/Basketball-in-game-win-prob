# College Basketball In-Game Win Probability

A nonlinear parametric in-game win probability model for NCAA men's
college basketball.

## Quick Start

### 1. Clone the repository

git clone https://github.com/ColinDonahoe119/Basketball-in-game-win-prob/tree/main

cd CBB_IGWP

### 2. Create a virtual environment

python -m venv venv

### 3. Activate the environment

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate

### 4. Install dependencies

pip install -r requirements.txt

## Running the Project

There are two ways to run the project, both are run from the in_game_win_prob.ipynb notebook.

### Option 1: Download Processed Data

This is the fastest option and allows you to skip data preparation.

Run:

download_processed()

Then run the following workflow:

1. Train the model
2. Evaluate the model
3. Generate analysis and visualizations

[Specific notebook instructions]

### Option 2: Start From Raw Data

To reproduce the data preparation process:

1. Run `download_raw()`
2. Run `data_prep`
3. Run `train_model`
4. Run `game_eval`
5. Run `model_eval`
6. Run `game_analysis`

## Running in Google Colab

[Your specific Colab instructions]

## Project Structure

CBB_IGWP/
├── src/
├── notebooks/
├── docs/
...

## Documentation

For the full methodology, data description, model formulation,
evaluation methodology, and results, see:

`docs/CBB_IGWP_Documentation.docx`
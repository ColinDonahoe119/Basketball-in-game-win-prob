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

## Running the Project

The project is designed to be run from the main notebook:

`docs/in_game_win_probability.ipynb`

After installing the required packages, open the notebook and follow the workflow in order.

### Data Options

At the beginning of the notebook, choose one of the following options:

**Option 1: Download Processed Data**

This is the fastest option and skips the data preparation step. Run the `download_processed()` function when prompted, then continue through the notebook workflow.

**Option 2: Start From Raw Data**

This option reproduces the data preparation process from the original raw data. Run `download_raw()` when prompted, then continue through the notebook workflow.

> **Note:** If using processed data, skip the data preparation section and continue with evaluation sections.
## Project Structure

CBB_IGWP/
│
├── docs/
│   ├── in_game_win_probability.ipynb
│   └── scripts/
│       ├── data_prep.py
│       ├── download_data.py
│       ├── game_analysis.py
│       ├── game_eval.py
│       ├── model_eval.py
│       ├── models.py
│       ├── train_model.py
│       └── win_probability_chart.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── data_prep.ipynb
├── requirements.txt
├── README.md
└── .gitignore

## Documentation

For the full methodology, data description, model formulation,
evaluation methodology, and results, see:

`docs/CBB_IGWP_Documentation.docx`

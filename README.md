# Project Structure


## **`config/`**
Contains configuration files to define reusable constants and paths used throughout the project.
- **`config.py`**: Centralized file that holds configuration parameters.

## **`data/`**
Directory to store datasets used in the project.
- **`data_house.csv`**: The original raw dataset containing information about houses (e.g., features like size, location, price, etc.).
- **`cleaned_data.csv`**: The cleaned and preprocessed dataset after applying transformations and feature engineering.

## **`models/`**
Folder to save trained machine learning models after training (`.pkl` files).

## **`notebook/`**
Contains a Jupyter notebook for analysis, data exploration, and model evaluation.
- **`housing_prices.ipynb`**: Main notebook where the project workflow is documented, including data exploration, preprocessing, model training and evaluation, and conclusions.

## **`output/`**
Directory to save graphs generated during the project.
- **Example files**: Graphs comparing predicted vs. actual prices or any exported analysis outputs.

## **`src/`**
Source code for helper functions and utilities that make the project modular and reusable.
- **`funs.py`**: Contains functions for preprocessing the data, training models, visualizing results and others.

## **`requirements.txt`**
Lists all the Python libraries required for the project. Use the following command to install them:  
```bash
pip install -r requirements.txt
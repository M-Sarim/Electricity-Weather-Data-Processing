# Electricity & Weather Data Processing

This repository provides a complete workflow for loading, preprocessing, and analyzing electricity demand data alongside weather data. The project includes:

- **Raw Data Handling** (Electricity and weather data processing)
- **Data Preprocessing** (Handling missing values, merging datasets, feature engineering)
- **Exploratory Data Analysis (EDA)** (Summary statistics, correlation analysis, visualization)
- **Time Series Analysis** (Trend analysis, stationarity testing, decomposition)
- **Outlier Detection** (Using IQR and Z-score methods)
- **Regression Modeling** (Predicting electricity demand based on weather conditions)
- **Data Export** (Processed datasets, summary statistics, correlation matrices)

---

## Project Structure

```md
Electricity-Weather-Data-Processing/
│── electricity_raw_data/       # Raw electricity demand data (JSON format)
│── weather_raw_data/           # Raw weather data (CSV format)
│── electricity_demand_pipeline.ipynb  # Jupyter Notebook for analysis
│── electricity_demand_pipeline.py     # Python script for analysis
│── final_cleaned_processed_data.csv  # Preprocessed dataset
│── README.md                   # Project documentation
```

### Dataset Details
- **electricity_raw_data/** → Contains JSON files with electricity demand data, structured as `response.data` with `period` (timestamp) and `value` (demand).
- **weather_raw_data/** → Contains CSV files with weather records, including at least `date` (timestamp) and `temperature_2m`.
- **processed_electricity_weather_data.csv** → Merged and cleaned dataset combining electricity and weather data.
- **correlation_matrix.csv** & **summary_statistics.csv** → Generated statistical insights.

---

## Installation Guide

### 1. Install Python
Ensure you have Python 3.7+ installed. If not, download it from [Python.org](https://www.python.org/downloads/).

### 2. Clone the Repository
Clone the project to your local machine using Git:

```bash
git clone https://github.com/M-Sarim/Electricity-Weather-Data-Processing.git
cd EDA_DataProcessing
```

### 3. Set Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 4. Install Dependencies
Install required libraries using pip:


```bash
pip install pandas numpy matplotlib seaborn scikit-learn rich scipy statsmodels
```

---

## Usage

### Running the Analysis
You can run the analysis using either the Jupyter Notebook or Python script:

#### Option 1: Jupyter Notebook
```bash
jupyter notebook electricity_demand_pipeline.ipynb
```

#### Option 2: Python Script
```bash
python electricity_demand_pipeline.py
```

---

## Key Features

### 1. Data Preprocessing
- Loads electricity and weather datasets.
- Handles missing values and merges datasets.
- Feature engineering for time-based analysis.

### 2. Exploratory Data Analysis
- Generates summary statistics.
- Computes and visualizes correlation matrices.
- Plots electricity demand trends.

### 3. Time Series Analysis
- Tests for stationarity.
- Applies seasonal decomposition.

### 4. Regression Modeling
- Builds a linear regression model to predict electricity demand.
- Uses a train-test split strategy.

### 5. Output Generation
- Saves processed data and statistical summaries for further analysis.

---

## Contribution
Feel free to contribute by:
- Improving the data preprocessing pipeline.
- Adding more statistical or machine-learning models.
- Enhancing visualizations and insights.

To contribute, fork the repository and submit a pull request with your improvements.

---

## License
This project is open-source and available under the MIT License.

---

## Contact
For any questions or suggestions, please reach out via GitHub Issues or email: **muhammad2004sarim@gmail.com**. 

---

Happy coding! 🚀

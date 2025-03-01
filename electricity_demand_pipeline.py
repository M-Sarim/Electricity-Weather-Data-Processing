import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.progress import track
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore, skew, kurtosis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Set global plotting style for aesthetics
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# Initialize Console for Logging
console = Console()


def load_electricity_data(electricity_folder: str) -> pd.DataFrame:
    """
    Load and merge electricity data from JSON files.
    """
    console.print("[yellow]Loading Electricity Data...[/yellow]")
    files = glob.glob(os.path.join(electricity_folder, "*.json"))
    df_list = []
    for file in track(files, description="Processing electricity JSON files..."):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "response" in data and "data" in data["response"]:
                df = pd.DataFrame(data["response"]["data"])
                if 'period' in df.columns and 'value' in df.columns:
                    df.rename(columns={'period': 'timestamp', 'value': 'electricity_demand'}, inplace=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df['electricity_demand'] = pd.to_numeric(df['electricity_demand'], errors='coerce')
                    df.dropna(subset=['timestamp'], inplace=True)
                    df_list.append(df)
                else:
                    console.print(f"[red]⚠ Skipping {file}: Missing required columns.[/red]")
            else:
                console.print(f"[red]⚠ Skipping {file}: Incorrect JSON structure.[/red]")
        except Exception as e:
            console.print(f"[red]❌ Error processing {file}: {e}[/red]")
    merged_electricity = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    console.print(f"[green]✅ Loaded {len(merged_electricity)} electricity records.[/green]")
    return merged_electricity


def load_weather_data(weather_folder: str) -> pd.DataFrame:
    """
    Load and merge weather data from CSV files.
    """
    console.print("[yellow]Loading Weather Data...[/yellow]")
    files = glob.glob(os.path.join(weather_folder, "*.csv"))
    df_list = []
    for file in track(files, description="Processing weather CSV files..."):
        try:
            df = pd.read_csv(file, encoding="utf-8")
            df_list.append(df)
        except Exception as e:
            console.print(f"[red]❌ Error processing {file}: {e}[/red]")
    merged_weather = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    if not merged_weather.empty:
        merged_weather.rename(columns={'date': 'timestamp'}, inplace=True)
        merged_weather['timestamp'] = pd.to_datetime(merged_weather['timestamp'], errors='coerce')
        merged_weather.dropna(subset=['timestamp'], inplace=True)
    console.print("[green]✅ Weather Data Processed Successfully![/green]")
    return merged_weather


def merge_datasets(electricity_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent timestamp formats and merge the electricity and weather data.
    """
    console.print("[yellow]Standardizing timestamps...[/yellow]")
    if 'timestamp' in electricity_df.columns:
        electricity_df['timestamp'] = electricity_df['timestamp'].dt.tz_localize(None)
    if 'timestamp' in weather_df.columns:
        weather_df['timestamp'] = weather_df['timestamp'].dt.tz_localize(None)
    merged = pd.merge(electricity_df, weather_df, on="timestamp", how="inner")
    console.print(f"[cyan]✅ Data Merged Successfully: {merged.shape[0]} records, {merged.shape[1]} features.[/cyan]")
    return merged


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the merged data: impute missing values, remove duplicates,
    and add time-based features.
    """
    console.rule("[bold blue]Step 2: Data Preprocessing[/bold blue]")
    console.print("[yellow]Analyzing missing data...[/yellow]")
    missing = df.isnull().sum() / len(df) * 100
    console.print("Missing Values (% per column):")
    console.print(missing)
    df.ffill(inplace=True)
    df.drop_duplicates(inplace=True)
    console.print("[green]✅ Missing values handled and duplicates removed![/green]")
    
    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    console.print("[green]✅ Data Preprocessed Successfully![/green]")
    return df


def perform_eda(df: pd.DataFrame):
    """
    Perform exploratory data analysis and generate aesthetically enhanced plots.
    """
    console.rule("[bold blue]Step 3: Exploratory Data Analysis (EDA)[/bold blue]")
    
    # Statistical Summary
    console.print("[yellow]Statistical Summary:[/yellow]")
    stats = df.describe().T
    stats['skewness'] = df.select_dtypes(include=[np.number]).apply(skew)
    stats['kurtosis'] = df.select_dtypes(include=[np.number]).apply(kurtosis)
    console.print(stats)
    
    # Time Series Plot
    plt.figure()
    plt.plot(df['timestamp'], df['electricity_demand'], color="mediumblue", linewidth=2, label="Electricity Demand")
    plt.xlabel('Time')
    plt.ylabel('Electricity Demand')
    plt.title('Electricity Demand Over Time')
    plt.xticks(rotation=45)
    if not df.empty:
        peak = df['electricity_demand'].idxmax()
        plt.annotate('Peak Demand',
                     xy=(df.loc[peak, 'timestamp'], df.loc[peak, 'electricity_demand']),
                     xytext=(df.loc[peak, 'timestamp'], df.loc[peak, 'electricity_demand'] * 1.05),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Univariate Plots
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    sns.histplot(df['electricity_demand'], kde=True, color="slateblue", edgecolor="black", ax=axs[0])
    axs[0].set_title("Histogram & Density")
    sns.boxplot(y=df['electricity_demand'], color="lightseagreen", ax=axs[1])
    axs[1].set_title("Boxplot")
    sns.kdeplot(df['electricity_demand'], fill=True, color="crimson", ax=axs[2])
    axs[2].set_title("KDE Density Plot")
    for ax in axs:
        ax.set_xlabel("Electricity Demand")
    plt.tight_layout()
    plt.show()
    
    # Correlation Heatmap
    plt.figure()
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    
    # Advanced Time Series Analysis: Seasonal Decomposition
    try:
        # Resample to daily frequency and fill missing values
        ts = df.set_index('timestamp')['electricity_demand'].resample('D').mean().ffill()
        decomposition = seasonal_decompose(ts, model='additive', period=7)
        fig = decomposition.plot()
        fig.set_size_inches(14, 10)
        plt.suptitle("Seasonal Decomposition of Electricity Demand", fontsize=18)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    except Exception as e:
        console.print(f"[red]❌ Seasonal Decomposition failed: {e}[/red]")
    
    # Augmented Dickey-Fuller Test
    try:
        adf_result = adfuller(ts)
        console.print("[yellow]Augmented Dickey-Fuller Test Results:[/yellow]")
        console.print(f"ADF Statistic: {adf_result[0]:.4f}")
        console.print(f"p-value: {adf_result[1]:.4f}")
        if adf_result[1] < 0.05:
            console.print("[green]The time series is likely stationary.[/green]")
        else:
            console.print("[red]The time series is likely non-stationary.[/red]")
    except Exception as e:
        console.print(f"[red]❌ ADF Test failed: {e}[/red]")
    
    console.print("[green]✅ EDA Completed Successfully![/green]")


def detect_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using IQR and Z-score methods and return a cleaned DataFrame.
    """
    console.rule("[bold blue]Step 4: Outlier Detection and Handling[/bold blue]")
    
    # Plot before outlier removal
    plt.figure()
    sns.histplot(df['electricity_demand'], kde=True, color="darkorange", edgecolor="black")
    plt.title("Electricity Demand Distribution (Before Outlier Removal)")
    plt.xlabel("Electricity Demand")
    plt.tight_layout()
    plt.show()
    
    # IQR-based filtering
    Q1 = df['electricity_demand'].quantile(0.25)
    Q3 = df['electricity_demand'].quantile(0.75)
    IQR = Q3 - Q1
    df_iqr = df[(df['electricity_demand'] >= Q1 - 1.5 * IQR) &
                (df['electricity_demand'] <= Q3 + 1.5 * IQR)]
    
    # Z-score filtering on numeric columns
    numeric_df = df_iqr.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_df))
    df_clean = df_iqr[(z_scores < 3).all(axis=1)]
    
    # Plot after outlier removal
    plt.figure()
    sns.histplot(df_clean['electricity_demand'], kde=True, color="mediumpurple", edgecolor="black")
    plt.title("Electricity Demand Distribution (After Outlier Removal)")
    plt.xlabel("Electricity Demand")
    plt.tight_layout()
    plt.show()
    
    console.print("[green]✅ Outliers detected and removed successfully![/green]")
    return df_clean


def regression_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform regression analysis on the cleaned dataset and return predictions.
    """
    console.rule("[bold blue]Step 5: Regression Modeling[/bold blue]")
    
    if 'temperature_2m' not in df.columns:
        console.print("[red]❌ ERROR: 'temperature_2m' column is missing from weather data![/red]")
        return pd.DataFrame()
    
    X = df[['hour', 'day', 'month', 'temperature_2m']]
    y = df['electricity_demand']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    console.print(f"✅ Model Trained! MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # Plot Actual vs Predicted
    plt.figure()
    plt.scatter(y_test, y_pred, color='mediumblue', alpha=0.6, edgecolor="black")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.title("Actual vs. Predicted Electricity Demand")
    plt.tight_layout()
    plt.show()
    
    # Residual analysis
    residuals = y_test - y_pred
    plt.figure()
    sns.histplot(residuals, kde=True, color="darkslategray", edgecolor="black")
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.scatter(y_pred, residuals, color="teal", alpha=0.6, edgecolor="black")
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Demand")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Predicted Values")
    plt.tight_layout()
    plt.show()
    
    console.print("[green]✅ Regression modeling and residual analysis completed successfully![/green]")
    
    predictions_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Residual": residuals
    })
    return predictions_df


def run_analysis():
    """
    Main function to run the complete analysis pipeline.
    """
    console.rule("[bold blue]Step 1: Data Loading & Integration[/bold blue]")
    
    # Define paths
    base_path = r"/Users/muhammadsarim/Desktop/22F-3688_BSE-6B_AssNO.2"
    elec_folder = os.path.join(base_path, "electricity_raw_data")
    weather_folder = os.path.join(base_path, "weather_raw_data")
    
    # Load datasets
    electricity_df = load_electricity_data(elec_folder)
    weather_df = load_weather_data(weather_folder)
    
    # Merge datasets
    merged_df = merge_datasets(electricity_df, weather_df)
    console.print(f"[blue]Initial merged data: {merged_df.shape[0]} records, {merged_df.shape[1]} features.[/blue]")
    
    # Preprocess data and add features
    processed_df = preprocess_data(merged_df)
    
    # Exploratory Data Analysis
    perform_eda(processed_df)
    
    # Outlier detection and cleaning
    clean_df = detect_and_handle_outliers(processed_df)
    
    # Remove the 'season' column before generating the final CSV
    if 'season' in clean_df.columns:
        clean_df = clean_df.drop(columns=['season'])
    
    # Generate final cleaned and processed CSV
    final_csv_path = os.path.join(base_path, "final_cleaned_processed_data.csv")
    clean_df.to_csv(final_csv_path, index=False)
    console.print(f"[green]Final cleaned and processed data saved at: {final_csv_path}[/green]")
    
    # Regression modeling
    predictions_df = regression_modeling(clean_df)
    if not predictions_df.empty:
        console.print("[green]Regression modeling completed and predictions obtained.[/green]")


if __name__ == "__main__":
    run_analysis()

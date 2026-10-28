# Cinema Audience Forecasting

A data science and machine learning project focused on forecasting daily cinema audience counts across multiple theater locations. Developed as part of a Kaggle competition, this project explores historical attendance, booking activity, theater characteristics, and temporal patterns to build and compare audience prediction models.

## Project Overview

Accurately forecasting cinema attendance can help theater operators understand demand, plan operations, and make better use of their resources.

This project uses historical cinema visit records and associated booking and theater metadata to explore audience behavior and generate predictions for future theater-show dates.

The workflow covers data cleaning, dataset integration, exploratory data analysis (EDA), feature preparation, statistical forecasting, machine learning model training, hyperparameter tuning, model comparison, and submission generation.

### Project Highlights

- Analyzed approximately **214,000 daily theater attendance records**.
- Worked with historical data from **826 theaters**.
- Integrated multiple datasets containing visit records, booking information, theater metadata, and date-related information.
- Investigated weekday and weekend attendance patterns, theater-level differences, monthly trends, booking behavior, and missing data.
- Developed and compared statistical forecasting approaches with Random Forest, Ridge Regression, XGBoost, and LightGBM.
- Applied RandomizedSearchCV for hyperparameter optimization.
- Generated predictions for **38,062 test records** in the required Kaggle submission format.

## Objectives

The primary objectives of this project were to:

1. Understand the factors and patterns associated with daily cinema attendance.
2. Clean and integrate datasets from different sources.
3. Identify temporal, geographic, and theater-level variations in audience counts.
4. Engineer useful features from booking information, dates, and theater metadata.
5. Establish statistical forecasting baselines.
6. Train and evaluate multiple machine learning algorithms.
7. Compare models using R² and Mean Absolute Error (MAE).
8. Generate a prediction file suitable for competition submission.

## Dataset Overview

The project uses datasets provided through the Kaggle competition environment. These datasets represent historical cinema visits, bookings, theater metadata, and the test records for which predictions are required.

### Input datasets

| Dataset | Purpose |
|---|---|
| `booknow_visits.csv` | Historical daily theater visits and audience counts. |
| `booknow_booking.csv` | Booking activity, including tickets booked for theater-show dates. |
| `cinePOS_booking.csv` | Point-of-sale booking information, including tickets sold. |
| `booknow_theaters.csv` | Theater metadata associated with BookNow theater identifiers. |
| `cinePOS_theaters.csv` | Theater information associated with CinePOS identifiers. |
| `movie_theater_id_relation.csv` | Mapping between CinePOS and BookNow theater identifiers. |
| `date_info.csv` | Date-related information used to analyze temporal patterns. |
| `sample_submission.csv` | Test identifiers and the required prediction output structure. |

The files are loaded from Kaggle's input directory. The original competition datasets are not included in this repository.

### Dataset characteristics

The notebook's processed historical dataset contains:

- **214,036 rows** after duplicate removal and dataset integration.
- **826 unique theaters**.
- A historical date range from January 1, 2023, to February 28, 2024.
- Daily audience counts as the prediction target.
- Booking-related fields, including tickets booked and tickets sold.
- Theater-related attributes, such as theater type and area.
- Temporal attributes, including show dates and days of the week.

The notebook also prepares a test dataset containing 38,062 records covering March 1, 2024, to April 22, 2024.

## Technology Stack

| Technology | Usage |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation, aggregation, and dataset integration |
| NumPy | Numerical computations and array operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization and correlation analysis |
| Scikit-learn | Model training, preprocessing, evaluation, and hyperparameter tuning |
| Random Forest | Tree-based regression model |
| Ridge Regression | Regularized linear regression |
| XGBoost | Gradient-boosted tree regression |
| LightGBM | Gradient-boosted decision tree regression |
| Jupyter Notebook | Interactive analysis and experimentation |
| Kaggle | Competition dataset access and prediction submission |

## Methodology

### 1. Data Loading

The notebook loads the available datasets from the Kaggle competition input directory.

The data includes historical visits, online booking information, point-of-sale ticket sales, theater metadata, date information, and the sample submission structure.

Initial inspection is performed to understand dataset dimensions and available columns.

### 2. Data Cleaning

The following preprocessing steps are performed:

- Removal of duplicate records from the input datasets.
- Conversion of date and datetime columns into appropriate datetime formats.
- Extraction of show dates from booking timestamps.
- Alignment of date formats across datasets.
- Preparation of theater identifiers for merging.
- Aggregation of booking activity at the theater-and-date level.

These steps establish consistent keys and date formats before integrating the datasets.

### 3. Dataset Integration

Booking data is aggregated by theater and show date.

CinePOS theater identifiers are mapped to the corresponding BookNow theater identifiers using the provided relationship dataset.

The historical visit records are then joined with booking aggregates, ticket sales, date information, and theater metadata.

The resulting processed dataset contains the audience target and the available explanatory variables for subsequent analysis.

The merged dataset is sorted by theater and show date to support time-based exploration and rolling calculations.

### 4. Exploratory Data Analysis (EDA)

Exploratory analysis is used to understand the distribution of audience counts and identify potentially useful forecasting signals.

The notebook investigates the following areas:

#### Audience distribution

- Examination of audience count distributions.
- Comparison of mean and median attendance.
- Identification of skewness and high-attendance outliers.

The analysis indicates that attendance is right-skewed, with many observations at lower or moderate attendance levels and a smaller number of high-attendance observations.

#### Missing data

Missing values are examined across booking and theater-related attributes.

The notebook identifies substantial missingness in ticket-related fields and theater metadata. These limitations are considered when preparing features for machine learning.

#### Day-of-week patterns

Audience counts are compared across the days of the week.

The analysis identifies a recurring weekly pattern, with weekends generally attracting larger audiences than weekdays.

#### Theater-level performance

Audience averages are compared across individual theaters to identify differences in attendance.

This analysis highlights variation between theater locations and suggests that theater-specific historical behavior can be useful for forecasting.

#### Theater types and areas

Attendance patterns are explored across theater types and geographic area categories.

The analysis shows differences in audience levels between areas, while average attendance across theater types is comparatively similar.

#### Temporal trends

Time-series visualizations and moving averages are used to investigate changes in audience counts over time.

Monthly statistics are calculated to compare average and total attendance.

#### Correlation analysis

Numerical variables are examined using a correlation matrix.

The notebook reports the following approximate correlations with audience count:

| Feature | Correlation |
|---|---:|
| `tickets_booked` | 0.453 |
| `tickets_sold` | 0.267 |
| `longitude` | 0.045 |
| `latitude` | 0.039 |

Ticket-related variables show stronger positive correlations than the geographic coordinates in this analysis. Correlation alone, however, does not establish causation or guarantee predictive performance.

### 5. Feature Engineering and Preprocessing

The project prepares features representing different aspects of theater attendance.

These include:

- **Temporal features:** day of the week and show date.
- **Theater features:** theater identifier, theater type, and theater area.
- **Booking features:** tickets booked and tickets sold.
- **Historical statistical features:** theater-specific weekday averages and recent rolling attendance averages.

For the machine learning workflow, numerical missing values are filled using their medians, while categorical missing values are filled using their modes.

Categorical features are converted into numerical representations using one-hot encoding.

The notebook uses a chronological, non-shuffled 80/20 train-validation split for the machine learning experiments.

### 6. Statistical Forecasting Models

Several statistical forecasting approaches are constructed using historical weekday patterns, theater averages, and recent rolling trends.

The weekday average captures recurring differences in attendance across the days of the week.

The rolling average captures more recent attendance behavior.

Fallback averages are used when a weekday-specific or rolling value is unavailable.

Four approaches are compared:

| Model | Approach |
|---|---|
| Statistical Model 1 | Equal weighting of weekday averages and rolling trends (50/50). |
| Statistical Model 2 | Greater emphasis on recent rolling trends (70/30). |
| Statistical Model 3 | Greater emphasis on weekday averages (70/30). |
| Statistical Model 4 | Adaptive weighting based on the component models' R² scores. |

The adaptive approach combines the weekday and rolling components using weights calculated from their respective R² scores.

This provides a simple baseline for evaluating how much predictive value can be obtained from historical attendance patterns without relying exclusively on a complex machine learning algorithm.

### 7. Machine Learning Models

The notebook also evaluates several regression algorithms.

#### Random Forest Regressor

A tree-based ensemble model is trained to capture nonlinear relationships between the selected features and audience counts.

#### Ridge Regression

Ridge Regression provides a regularized linear baseline for examining the predictive value of the encoded feature set.

#### XGBoost

XGBoost is evaluated as a gradient-boosting approach for learning relationships between theater, booking, and temporal attributes.

#### LightGBM

LightGBM is included as another gradient-boosting algorithm for regression.

The machine learning experiments use a chronological training and validation split, with model performance assessed on the held-out portion of the processed data.

### 8. Hyperparameter Tuning

RandomizedSearchCV is used to explore hyperparameter configurations for the machine learning model selected by the notebook's initial R² comparison.

The search uses:

- 20 randomly sampled hyperparameter configurations.
- Three-fold cross-validation.
- Negative Mean Squared Error as the search scoring metric.
- A fixed random seed for reproducibility.

For the XGBoost model selected in the initial comparison, the notebook explores parameters including:

- Number of estimators.
- Maximum tree depth.
- Learning rate.
- Row subsampling.
- Feature subsampling.

The tuned model is subsequently evaluated using R² and MAE.

### 9. Model Evaluation and Comparison

The notebook compares statistical and machine learning approaches using two regression metrics.

**R² score**

R² measures how well model predictions explain variation in the target relative to a mean-based baseline. Higher values generally indicate better fit.

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between predicted and actual audience counts. Lower values indicate smaller prediction errors.

The comparison helps identify which approach performs best under the evaluation procedure implemented in the notebook.

### 10. Test Data Preparation

The sample submission file provides the identifiers for the test observations.

The notebook extracts the theater identifier and show date from each test identifier, converts the date into datetime format, and joins the test records with theater metadata and date information.

Historical weekday averages, theater averages, and recent trend information are then used to prepare the statistical predictions.

### 11. Prediction Generation

After comparing the candidate models, the notebook selects the best-performing approach according to its overall R² comparison.

The adaptive statistical model is selected in the recorded notebook results.

Predictions are generated for the test records, negative predictions are clipped to zero, and values are rounded to integers.

### 12. Submission File Creation

The final predictions are saved to `submission.csv`.

The output contains the two required columns:

| Column | Description |
|---|---|
| `ID` | Original test-record identifier. |
| `audience_count` | Predicted daily audience count. |

The notebook verifies that the generated submission contains 38,062 rows and matches the expected column structure.

## Results

The following table reproduces the recorded model-comparison results from the notebook.

| Model | R² | MAE |
|---|---:|---:|
| Statistical Model 1 (50/50) | 0.6022 | 13.75 |
| Statistical Model 2 (70/30) | 0.5903 | 14.03 |
| Statistical Model 3 (30/70) | 0.5951 | 13.79 |
| Statistical Model 4 (Adaptive) | **0.6023** | **13.75** |
| Random Forest | 0.0653 | 22.47 |
| Ridge Regression | 0.0581 | 22.54 |
| XGBoost | 0.0693 | 21.96 |
| LightGBM | 0.0693 | 21.81 |
| Tuned XGBoost | 0.0680 | 22.12 |

*Note: These are the metrics recorded by the notebook's implemented evaluation workflow. They should not be interpreted as independently verified competition leaderboard scores.*

### Key Findings

- Historical weekday patterns and rolling attendance trends provide useful baseline signals.
- Weekend attendance is higher than weekday attendance in the exploratory analysis.
- Theater-level differences contribute to variation in audience counts.
- Ticket booking and sales fields have missing values that require careful preprocessing.
- The statistical forecasting approaches outperform the tested machine learning models under the notebook's recorded comparison.
- Hyperparameter tuning does not improve the recorded XGBoost R² score.
- The adaptive statistical model is selected for generating the final predictions.

### Evaluation Considerations

The statistical model metrics in this notebook are calculated on the same historical dataset used to construct the weekday and theater averages and rolling features. Consequently, those metrics are not an independent out-of-sample evaluation of the statistical forecasting approach.

A stronger evaluation would generate historical predictions using a strictly forward-in-time validation period, ensuring that each prediction uses only information available before its show date. This would also help prevent information leakage from future observations into historical averages and rolling features.

## Repository Structure

The repository is intended to contain the following files:

```text
Cinema-Audience-Forecasting/
├── README.md
└── 22f2000552-notebook-t32025 (1).ipynb
```

The notebook contains the data preparation workflow, exploratory analysis, model experiments, evaluation, and submission-generation code.

The original Kaggle datasets are not included in the repository.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python environment.
- Jupyter Notebook or JupyterLab.
- Access to the required Kaggle competition datasets.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Cinema-Audience-Forecasting.git
cd Cinema-Audience-Forecasting
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
```

### 4. Obtain the competition datasets

Download the required datasets from the relevant Kaggle competition and make them available in your execution environment.

The notebook currently references Kaggle input paths under:

```text
/kaggle/input/Cinema_Audience_Forecasting_challenge/
```

These paths are specific to the Kaggle environment. When running the notebook locally, update the file paths to match the location of the downloaded datasets.

### 5. Run the notebook

```bash
jupyter notebook
```

Open the `.ipynb` file and execute the cells in order.

**Important:** The notebook depends on the original competition datasets. It will not run end to end until those files are available at the expected paths or the paths are updated.

## Future Improvements

Potential improvements to the project include:

- Implementing a strictly time-based validation strategy for all model families.
- Preventing target leakage in rolling and historical aggregate features.
- Comparing predictions against a simple theater-level or global-mean baseline.
- Investigating missing booking and theater metadata in greater depth.
- Exploring additional date-derived features, holidays, and seasonal indicators where the source data supports them.
- Testing lagged attendance features calculated using past observations only.
- Improving categorical feature handling for theater identifiers and area attributes.
- Evaluating performance using the competition's official scoring metric and leaderboard.
- Recording reproducible experiment configurations and final model artifacts.

## Conclusion

This project demonstrates an end-to-end workflow for cinema audience forecasting, from raw dataset integration and exploratory data analysis to statistical modeling, machine learning experimentation, model comparison, and prediction-file generation.

An important finding from the recorded experiments is that relatively simple statistical forecasting methods based on weekday behavior and recent trends outperform the tested machine learning models under the notebook's current evaluation procedure.

The project also highlights the importance of careful feature engineering, data-quality analysis, time-aware validation, and leakage prevention when developing forecasting systems.

## Author

**Parth Chitranshi**

BS in Data Science and Applications, IIT Madras

### Project Links

- **GitHub Repository:** Add your repository URL after creating it.
- **Kaggle Competition:** Add the official competition URL if you want to link to it.

---

*This repository contains a downloadable version of the project notebook for review, reproducibility, and portfolio presentation. Dataset access remains subject to the original competition's terms and availability.*

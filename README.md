# House Price Prediction with Linear Regression

This project trains a Linear Regression model to predict house prices from the `Housing.csv` dataset.

> **Important:** On this dataset the model does not predict prices. The test R² is about **-0.007**, which means it performs no better than always guessing the average price. See [Results](#results) and [Why the model performs poorly](#why-the-model-performs-poorly).

## Dataset

`Housing.csv` has 2,000 rows and 10 columns. It has no missing values or duplicate rows.

| Column | Type | Description |
|---|---|---|
| `Id` | int | Row number (not used as a feature) |
| `Area` | int | Floor area (501 to 4,999) |
| `Bedrooms` | int | Number of bedrooms (1 to 5) |
| `Bathrooms` | int | Number of bathrooms (1 to 4) |
| `Floors` | int | Number of floors (1 to 3) |
| `YearBuilt` | int | Year built (1900 to 2023) |
| `Location` | categorical | Downtown, Suburban, Rural, or Urban |
| `Condition` | categorical | Excellent, Good, Fair, or Poor |
| `Garage` | categorical | Yes or No |
| `Price` | int | **Target:** sale price (about $50,000 to $1,000,000) |

## Requirements

- Python 3.8 or newer
- pandas
- numpy
- matplotlib
- scikit-learn

Install them with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Setup

1. Put `Housing.csv` in a location the script can read. In Google Colab, upload it to `/content/sample_data/` or update `DATA_PATH` to match where the file is.
2. Set `DATA_PATH` near the top of the script to the full path of the file:

   ```python
   DATA_PATH = "/content/sample_data/Housing.csv"  # or your own path
   ```

## Usage

Run the script from a terminal:

```bash
python house_price.py
```

Or run it cell by cell in a Jupyter or Colab notebook.

## What the script does

1. **Loads the data** from `DATA_PATH`.
2. **Splits the data** into 80% training and 20% test sets, using `random_state=42` so results are repeatable.
3. **Preprocesses the features** with a `ColumnTransformer`:
   - Numeric columns: median imputation, then standard scaling.
   - Categorical columns: most-frequent imputation, then one-hot encoding (the first category is dropped to avoid redundancy).
4. **Builds a pipeline** that combines preprocessing and `LinearRegression`. Fitting happens on training data only, which prevents data leakage.
5. **Evaluates the model**:
   - 5-fold cross-validated R² on the training set.
   - MAE, RMSE, and R² on the test set.
6. **Prints the coefficients** of each feature, sorted by absolute size.
7. **Saves and displays a plot** of predicted vs. actual prices on the test set, as `linear_regression_pred_vs_actual.png`.
8. **Predicts the price** of a sample house.

## Results

| Metric | Value |
|---|---|
| 5-fold CV R² (mean) | -0.009 |
| Test MAE | about $243,000 |
| Test RMSE | about $280,000 |
| Test R² | -0.007 |

The average house price in the dataset is about $538,000. An error of $243,000 on average is large compared with that, so the model's predictions are not useful.

## Why the model performs poorly

- **The features have almost no linear relationship with price.** The correlations between price and each numeric feature are all below 0.06 in absolute value.
- **Average price barely changes across categories.** For example, the mean price is about $519,000 in Urban areas and $557,000 in Suburban areas, and it is almost the same for each Condition and Garage value.
- **The dataset may be synthetic.** Real housing data usually shows a clear link between area and price. This file does not. If it is a practice or generated dataset, this result is what you should expect.

Because R² is near zero, the coefficients and the sample prediction in the script should not be treated as meaningful.

## Output files

| File | Description |
|---|---|
| `linear_regression_pred_vs_actual.png` | Predicted vs. actual prices on the test set |

## Next steps

- Use a real housing dataset, such as the Kaggle House Prices or Ames Housing data, where price depends more strongly on the features.
- Try non-linear models such as Random Forest or Gradient Boosting, which can find patterns that linear regression misses.
- Check how the data was generated before drawing conclusions from the model.

## Known issue in the current script

The script calls `os.path.join` when saving the plot, but the uploaded version doesn't import `os`. Running it will raise `NameError: name 'os' is not defined` after the metrics print. Add this line at the top of the script:

```python
import os
```

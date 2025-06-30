# Fiber-Reinforced Soil CBR Prediction Using Random Forest Regression

## Overview

This project evaluates the effect of natural fibers (stem and root fibers) on soil strength, specifically focusing on the California Bearing Ratio (CBR) values used for pavement subgrade design. The study involves experimental data on soil physical properties and CBR tests, combined with a Random Forest Regression machine learning model to predict and optimize soil strength based on fiber content and type.

The model accurately predicts CBR values and identifies optimal fiber dosages for maximum soil reinforcement.

---

## Dataset

The dataset includes these key features and target variable:

| Feature           | Description                                  |
|-------------------|----------------------------------------------|
| Fiber_Content (%)  | Percentage of fiber added to soil             |
| Fiber_Type        | Type of fiber (Stem = 0, Root = 1)            |
| Dry_Density (g/cc) | Maximum dry density of fiber-reinforced soil |
| Moisture_Content (%) | Optimum moisture content for compaction      |
| CBR_Value (%)      | California Bearing Ratio (strength indicator) |

The sample dataset contains fiber contents ranging from 0.25% to 1.25% for stem fibers and 0.25% to 1.0% for root fibers, with corresponding soil compaction and CBR test results.

---

## Repository Contents

- `random_forest_cbr_prediction.py`: Python script to train and evaluate a Random Forest Regression model on fiber-reinforced soil data.
- `actual_vs_predicted_cbr.png`: Plot comparing actual and predicted CBR values.
- `feature_importance.png`: Feature importance ranking from the Random Forest model.
- `cbr_vs_fiber_content.png`: Plot of CBR values against fiber content for stem and root fibers.

---

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn

Install required packages via pip:

```bash
pip install numpy pandas matplotlib scikit-learn

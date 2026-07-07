# Road Traffic Accident Severity Prediction

Predicts the severity of a road traffic accident (Slight/Serious/Fatal Injury) from data regarding: accident, vehicle, driver and road condition data, using a Random Forest classifier.

## Dataset

`RTA Dataset.csv` — 12,316 records, 32 columns covering driver-related information, vehicle type, road type, weather conditions and casualty details.

Target column: `Accident_severity`

Class distribution is imbalanced:
- Slight Injury: 84.6%
- Serious Injury: 14.2%
- Fatal injury: 1.3%

## Pipeline

1. **Missing value imputation** — mode imputation on 16 columns with missing data (~36% missing for `Defect_of_vehicle`).
2. **Type conversion** — `Time` parsed and converted to seconds-from-midnight.
3. **Encoding** — one-hot encoding of categorical features.
4. **Class imbalance** — handled with SMOTE oversampling (applied to the training split only, after train/test split, to avoid leakage into the test set).
5. **Model** — `RandomForestClassifier` (scikit-learn).

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

## Usage

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook iai.ipynb
```

## Results

Due to class imbalance — always predicting "Slight Injury" alone gets ~84.6% accuracy. On a proper test set (SMOTE applied only to training data, `Casualty_severity` excluded as a feature since it is an almost-duplicate of the target and causes data leaking):

- **Accuracy: 84.2%**
- Slight Injury — precision 0.85, recall 0.99, F1 0.91
- Serious Injury — precision 0.34, recall 0.03, F1 0.06
- Fatal injury — precision 0.00, recall 0.00, F1 0.00

The model essentially predicts the majority class and fails to identify Serious or Fatal accidents.

## Known limitations / Future work

- `Casualty_severity` is excluded as a feature — it is almost the same as the taget feature and causes data leakage if included.

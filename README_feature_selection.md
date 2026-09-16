# Prostate Cancer Radiomics Feature Selection

## Project Overview

This project implements a streamlined prostate cancer radiomics feature selection pipeline. It focuses solely on feature selection using Spearman correlation filtering and recursive feature elimination (RFE).

## Workflow Description

### 1. Data Loading and Preprocessing
- Load clinical data, conventional radiomics features, and habitat heterogeneity indices
- Merge data and handle missing values
- Standardize features (StandardScaler)
- 80/20 stratified split into training and test sets

### 2. Spearman Correlation Filtering
- Calculate Spearman correlation coefficient matrix
- Remove highly correlated feature pairs (|rs| > 0.75)
- Generate correlation heatmap and feature pair list

### 3. Recursive Feature Elimination (RFE) Feature Selection
- Use Random Forest as base estimator
- 5-fold stratified cross-validation to select optimal feature subset
- Generate feature ranking and selection performance curve

### 4. Summary and Output
- Generate feature selection summary report
- Save selected features and feature rankings
- Output visualization of feature selection performance

## File Structure

```
feature_selection_results/
├── data_info.json                 # Data split information
├── spearman_correlation_matrix_*.csv  # Spearman correlation matrix
├── spearman_correlation_heatmap_*.pdf # Correlation heatmap
├── high_correlation_pairs_*.csv   # Highly correlated feature pairs
├── filtered_features.json         # Filtered feature list
├── rfe_performance_curve_*.pdf    # RFE performance curve
├── feature_ranking_rfe_*.csv      # RFE feature ranking
├── rfe_selected_features.json     # RFE selected features
└── feature_selection_summary.json # Summary of feature selection results
```

## Usage

### 1. Data Preparation

Prepare three CSV files:

#### clinic_data.csv (Clinical Data)
```csv
ID,label,PI-RADS,age,psa,...
P001,1,4,65,5.2,...
P002,0,3,72,3.8,...
```

#### radiomics_features.csv (Conventional Radiomics Features)
```csv
ID,T2WI_glcm_contrast,T2WI_glcm_correlation,T1WIC_firstorder_mean,...
P001,12.5,0.82,102.3,...
P002,8.7,0.91,98.6,...
```

#### habitat_heterogeneity_index.csv (Habitat Heterogeneity Index)
```csv
ID,T2WI_HI_entropy,T2WI_HI_uniformity,T1WIC_HI_entropy,...
P001,2.34,0.45,2.12,...
P002,1.98,0.52,1.89,...
```

### 2. Run the Script

```python
# Import the feature selector
from feature_selection_prostate import ProstateFeatureSelector

# Create selector instance
selector = ProstateFeatureSelector(
    base_dir='./your_data_directory',
    random_state=42
)

# Run the feature selection pipeline
results = selector.run_pipeline(
    clinic_file='clinic_data.csv',
    radiomics_file='radiomics_features.csv',
    hi_file='habitat_heterogeneity_index.csv',
    spearman_threshold=0.75,
    n_features_to_select=10,
    cv_folds=5
)
```

### 3. View Results

- All output files are saved in the `feature_selection_results/` directory
- Summary report is saved as `feature_selection_summary.json`
- Selected features are in `rfe_selected_features.json`

## Parameters

### ProstateFeatureSelector Parameters
- `base_dir`: Base directory path
- `random_state`: Random seed (default: 42)

### run_pipeline Parameters
- `clinic_file`: Clinical data file path
- `radiomics_file`: Conventional radiomics features file path
- `hi_file`: Habitat heterogeneity index file path
- `spearman_threshold`: Spearman correlation threshold (default: 0.75)
- `n_features_to_select`: Number of features to select with RFE (default: 10)
- `cv_folds`: Number of cross-validation folds (default: 5)

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Installation

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

## Output Files

### Main Outputs
1. **Data Information**: `data_info.json` - Data split and feature information
2. **Correlation Analysis**: Correlation matrix, heatmap, highly correlated feature pairs
3. **Feature Selection**: RFE performance curve, feature ranking, selected feature list
4. **Summary Report**: `feature_selection_summary.json` - Complete analysis summary

### Feature Selection Summary
The summary includes:
- Total samples and train/test split
- Original feature count
- Features after Spearman filtering
- Features after RFE selection
- Best cross-validation AUC
- List of selected features

## Notes

1. **Data Quality**: Ensure input data has no missing values or handle them appropriately
2. **Feature Naming**: T2WI and T1WIC features should follow the naming convention
3. **Randomness**: Set fixed random_state for reproducible results
4. **Computational Resources**: RFE with cross-validation may take time for large datasets
5. **Feature Interpretation**: Selected features should be interpreted in clinical context

## License

MIT License
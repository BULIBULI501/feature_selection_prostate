# -*- coding: utf-8 -*-
"""
Prostate Cancer Radiomics Feature Selection

Feature selection pipeline based on Spearman correlation filtering + 
Recursive Feature Elimination (RFE) for prostate cancer radiomics.

IMPORTANT: All feature selection is performed on the training set only.

Author: Stark
Date: 2026-09-16
Version: 2.0 (Simplified - Feature Selection Only)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42


class ProstateFeatureSelector:
    """
    Prostate Cancer Radiomics Feature Selector
    
    Workflow:
    1. Data loading and preprocessing
    2. Spearman correlation filtering (|rs| > 0.75)
    3. Recursive Feature Elimination (RFE) feature selection
    """
    
    def __init__(self, base_dir=None, random_state=42):
        """
        Initialize the feature selector.
        
        Parameters:
        -----------
        base_dir : str
            Base directory path
        random_state : int
            Random seed for reproducibility
        """
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir = base_dir
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Create output directory
        self.out_dir = os.path.join(base_dir, 'feature_selection_results')
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Record execution time
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("=" * 80)
        print("Prostate Cancer Radiomics Feature Selection")
        print("=" * 80)
        print(f"Execution time: {self.current_time}")
        print(f"Random seed: {random_state}")
        print(f"Output directory: {self.out_dir}")
    
    def load_data(self, clinic_file, radiomics_file, hi_file, label_col='label', id_col='ID'):
        """
        Load clinical data, conventional radiomics features, and habitat heterogeneity indices.
        
        Parameters:
        -----------
        clinic_file : str
            Path to clinical data file (CSV)
        radiomics_file : str
            Path to conventional radiomics features file (CSV)
        hi_file : str
            Path to habitat heterogeneity index file (CSV)
        label_col : str
            Label column name
        id_col : str
            ID column name
            
        Returns:
        --------
        dict : Dictionary containing training and test sets
        """
        print("\n[Step 1] Data loading and preprocessing...")
        
        # Load data
        df_clinic = pd.read_csv(clinic_file, encoding='utf-8-sig')
        df_radiomics = pd.read_csv(radiomics_file, encoding='utf-8-sig')
        df_hi = pd.read_csv(hi_file, encoding='utf-8-sig')
        
        print(f"  Clinical data: {len(df_clinic)} rows, columns: {list(df_clinic.columns[:5])}...")
        print(f"  Conventional radiomics: {len(df_radiomics)} rows, {len(df_radiomics.columns)} features")
        print(f"  Habitat heterogeneity index: {len(df_hi)} rows, {len(df_hi.columns)} features")
        
        # Merge data
        df_merged = pd.merge(
            df_clinic[[id_col, label_col]],
            df_radiomics,
            on=id_col,
            how='inner'
        )
        df_merged = pd.merge(
            df_merged,
            df_hi,
            on=id_col,
            how='inner'
        )
        
        print(f"  Merged data: {len(df_merged)} rows")
        
        # Separate features and labels
        X = df_merged.drop([id_col, label_col], axis=1)
        y = df_merged[label_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split into training and test sets (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"  Training set: {len(X_train)} samples (positive: {sum(y_train==1)}, negative: {sum(y_train==0)})")
        print(f"  Test set: {len(X_test)} samples (positive: {sum(y_test==1)}, negative: {sum(y_test==0)})")
        
        # Save data split information
        data_info = {
            'total_samples': len(X_scaled),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_train': int(sum(y_train==1)),
            'negative_train': int(sum(y_train==0)),
            'positive_test': int(sum(y_test==1)),
            'negative_test': int(sum(y_test==0)),
            'feature_names': list(X.columns),
            'scaler_params': {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
        }
        
        with open(os.path.join(self.out_dir, 'data_info.json'), 'w') as f:
            json.dump(data_info, f, indent=2)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'data_info': data_info
        }
    
    def spearman_filter(self, X_train, threshold=0.75):
        """
        Spearman correlation filtering to remove highly correlated features.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training set features
        threshold : float
            Correlation threshold, default 0.75
            
        Returns:
        --------
        pd.DataFrame : Filtered features
        """
        print(f"\n[Step 2] Spearman correlation filtering (|rs| > {threshold})...")
        
        # Calculate Spearman correlation matrix
        corr_matrix = X_train.corr(method='spearman')
        
        # Identify highly correlated feature pairs
        high_corr_pairs = []
        n_features = len(corr_matrix.columns)
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        print(f"  Found {len(high_corr_pairs)} highly correlated feature pairs (|rs| > {threshold})")
        
        # Save correlation matrix
        corr_file = os.path.join(self.out_dir, f'spearman_correlation_matrix_{self.current_time}.csv')
        corr_matrix.to_csv(corr_file, encoding='utf-8-sig')
        print(f"  Correlation matrix saved: {os.path.basename(corr_file)}")
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=False, 
            cmap='RdBu_r', 
            center=0,
            square=True, 
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Spearman Correlation Matrix', fontsize=14)
        plt.tight_layout()
        
        corr_fig = os.path.join(self.out_dir, f'spearman_correlation_heatmap_{self.current_time}.pdf')
        plt.savefig(corr_fig, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Correlation heatmap saved: {os.path.basename(corr_fig)}")
        
        # Save highly correlated feature pairs
        if high_corr_pairs:
            df_high_corr = pd.DataFrame(high_corr_pairs)
            high_corr_file = os.path.join(self.out_dir, f'high_correlation_pairs_{self.current_time}.csv')
            df_high_corr.to_csv(high_corr_file, index=False, encoding='utf-8-sig')
            print(f"  Highly correlated feature pairs saved: {os.path.basename(high_corr_file)}")
        
        # Remove highly correlated features (keep the first feature in each pair)
        features_to_remove = set()
        for pair in high_corr_pairs:
            # Remove the second feature (typically keep the first)
            features_to_remove.add(pair['feature2'])
        
        X_filtered = X_train.drop(columns=list(features_to_remove))
        
        print(f"  Features before filtering: {n_features}")
        print(f"  Features after filtering: {len(X_filtered.columns)}")
        print(f"  Features removed: {len(features_to_remove)}")
        
        # Save filtered feature list
        filtered_features = list(X_filtered.columns)
        with open(os.path.join(self.out_dir, 'filtered_features.json'), 'w') as f:
            json.dump(filtered_features, f, indent=2)
        
        return X_filtered, features_to_remove
    
    def rfe_feature_selection(self, X_train, y_train, n_features_to_select=10, cv=5):
        """
        Recursive Feature Elimination (RFE) feature selection.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training set features (filtered)
        y_train : pd.Series
            Training set labels
        n_features_to_select : int
            Number of features to select
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : RFE feature selection results
        """
        print(f"\n[Step 3] Recursive Feature Elimination (RFE) feature selection...")
        
        # Use Random Forest as base estimator
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # RFE with cross-validation
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            min_features_to_select=min(5, n_features_to_select),
            n_jobs=-1
        )
        
        selector = selector.fit(X_train, y_train)
        
        # Get selected features
        selected_features = X_train.columns[selector.support_].tolist()
        feature_ranking = pd.DataFrame({
            'feature': X_train.columns,
            'ranking': selector.ranking_,
            'selected': selector.support_
        }).sort_values('ranking')
        
        print(f"  RFE selected features: {len(selected_features)}")
        print(f"  Best cross-validation AUC: {selector.cv_results_['mean_test_score'].max():.4f}")
        
        # Plot performance curve during feature selection
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1),
                 selector.cv_results_['mean_test_score'], marker='o', markersize=4)
        plt.fill_between(range(1, len(selector.cv_results_['mean_test_score']) + 1),
                         selector.cv_results_['mean_test_score'] - selector.cv_results_['std_test_score'],
                         selector.cv_results_['mean_test_score'] + selector.cv_results_['std_test_score'],
                         alpha=0.2)
        plt.xlabel('Number of Features')
        plt.ylabel('Cross-Validation AUC')
        plt.title('RFE Feature Selection Performance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        rfe_fig = os.path.join(self.out_dir, f'rfe_performance_curve_{self.current_time}.pdf')
        plt.savefig(rfe_fig, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  RFE performance curve saved: {os.path.basename(rfe_fig)}")
        
        # Save feature ranking
        ranking_file = os.path.join(self.out_dir, f'feature_ranking_rfe_{self.current_time}.csv')
        feature_ranking.to_csv(ranking_file, index=False, encoding='utf-8-sig')
        print(f"  Feature ranking saved: {os.path.basename(ranking_file)}")
        
        # Save selected features
        with open(os.path.join(self.out_dir, 'rfe_selected_features.json'), 'w') as f:
            json.dump(selected_features, f, indent=2)
        
        return {
            'selected_features': selected_features,
            'feature_ranking': feature_ranking,
            'cv_score': selector.cv_results_['mean_test_score'].max(),
            'selector': selector
        }
    
    def run_pipeline(self, clinic_file, radiomics_file, hi_file, 
                    spearman_threshold=0.75, n_features_to_select=10, cv_folds=5):
        """
        Run the complete feature selection pipeline.
        
        Parameters:
        -----------
        clinic_file : str
            Path to clinical data file
        radiomics_file : str
            Path to conventional radiomics features file
        hi_file : str
            Path to habitat heterogeneity index file
        spearman_threshold : float
            Spearman correlation threshold
        n_features_to_select : int
            Number of features to select with RFE
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Complete analysis results
        """
        print("\n" + "=" * 80)
        print("Starting feature selection pipeline...")
        print("=" * 80)
        
        # Step 1: Data loading and preprocessing
        data = self.load_data(clinic_file, radiomics_file, hi_file)
        
        # Step 2: Spearman correlation filtering
        X_filtered, removed_features = self.spearman_filter(
            data['X_train'], 
            threshold=spearman_threshold
        )
        
        # Step 3: RFE feature selection
        rfe_results = self.rfe_feature_selection(
            X_filtered, 
            data['y_train'],
            n_features_to_select=n_features_to_select,
            cv=cv_folds
        )
        
        # Generate summary report
        self.generate_summary(data, removed_features, rfe_results)
        
        print("\n" + "=" * 80)
        print("Feature selection completed!")
        print(f"All results saved to: {self.out_dir}")
        print("=" * 80)
        
        return {
            'data': data,
            'spearman_results': (X_filtered, removed_features),
            'rfe_results': rfe_results
        }
    
    def generate_summary(self, data, removed_features, rfe_results):
        """Generate a summary report."""
        print("\n[Summary]")
        
        summary = {
            'total_samples': data['data_info']['total_samples'],
            'train_samples': data['data_info']['train_samples'],
            'test_samples': data['data_info']['test_samples'],
            'original_features': len(data['feature_names']),
            'features_after_spearman': len(data['feature_names']) - len(removed_features),
            'features_after_rfe': len(rfe_results['selected_features']),
            'best_cv_auc': rfe_results['cv_score'],
            'selected_features': rfe_results['selected_features']
        }
        
        with open(os.path.join(self.out_dir, 'feature_selection_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Samples: {summary['total_samples']} total, {summary['train_samples']} train, {summary['test_samples']} test")
        print(f"  Features: {summary['original_features']} → {summary['features_after_spearman']} (Spearman) → {summary['features_after_rfe']} (RFE)")
        print(f"  Best CV AUC: {summary['best_cv_auc']:.4f}")
        print(f"  Selected features: {summary['selected_features']}")


def main():
    """Main function: Example usage"""
    # Set base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create feature selector
    selector = ProstateFeatureSelector(
        base_dir=base_dir,
        random_state=42
    )
    
    # Example file paths
    clinic_file = os.path.join(base_dir, 'data', 'clinic_data.csv')
    radiomics_file = os.path.join(base_dir, 'data', 'radiomics_features.csv')
    hi_file = os.path.join(base_dir, 'data', 'habitat_heterogeneity_index.csv')
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [clinic_file, radiomics_file, hi_file]):
        print("\nExample data files not found. Please prepare the following files:")
        print(f"1. Clinical data file: {clinic_file}")
        print(f"2. Conventional radiomics features file: {radiomics_file}")
        print(f"3. Habitat heterogeneity index file: {hi_file}")
        print("\nFile format requirements:")
        print("- CSV format, UTF-8 encoding")
        print("- Contains ID column for merging")
        print("- Clinical data contains label column (0=non-cancer, 1=cancer)")
        print("- Radiomics features file contains numerical feature columns")
        
        # Create example data directory
        os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
        
        # Generate example data format description
        example_readme = """
# Data File Format Description

## 1. clinic_data.csv (Clinical Data)
- ID: Unique patient identifier
- label: Label (0=non-prostate cancer, 1=prostate cancer)
- PI-RADS: PI-RADS score (optional)
- Other clinical variables

## 2. radiomics_features.csv (Conventional Radiomics Features)
- ID: Unique patient identifier
- T2WI_*: T2-weighted imaging features (starting with T2)
- T1WIC_*: Contrast-enhanced T1-weighted imaging features (starting with T1)
- Other radiomics features

## 3. habitat_heterogeneity_index.csv (Habitat Heterogeneity Index)
- ID: Unique patient identifier
- T2WI_HI_*: T2-weighted imaging habitat heterogeneity index (containing HI)
- T1WIC_HI_*: Contrast-enhanced T1-weighted imaging habitat heterogeneity index (containing HI)
- Other heterogeneity indices

## Feature Naming Convention
- Features starting with T2WI belong to T2-weighted imaging model
- Features starting with T1WIC belong to contrast-enhanced T1-weighted imaging model
- Features containing "HI" belong to habitat heterogeneity index model
- Features not containing "HI" belong to conventional radiomics model
"""
        
        with open(os.path.join(base_dir, 'data', 'README_data_format.txt'), 'w', encoding='utf-8') as f:
            f.write(example_readme)
        
        print(f"\nData format description saved to: {os.path.join(base_dir, 'data', 'README_data_format.txt')}")
        return
    
    # Run complete pipeline
    results = selector.run_pipeline(
        clinic_file=clinic_file,
        radiomics_file=radiomics_file,
        hi_file=hi_file,
        spearman_threshold=0.75,
        n_features_to_select=10,
        cv_folds=5
    )
    
    print("\nPipeline execution completed!")
    print(f"Please check output directory: {selector.out_dir}")


if __name__ == "__main__":
    main()
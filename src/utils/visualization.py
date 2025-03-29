"""
Visualization utilities for ML pipeline.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def plot_feature_importances(importances, feature_names, n_top=20, figsize=(10, 6), 
                            output_file=None, title="Feature Importances"):
    """
    Plot feature importances as a bar chart.
    
    Parameters
    ----------
    importances : array-like
        Feature importance scores.
    feature_names : array-like
        Names of features.
    n_top : int, default=20
        Number of top features to display.
    figsize : tuple, default=(10, 6)
        Figure size.
    output_file : str or Path, default=None
        If provided, save the plot to this file.
    title : str, default="Feature Importances"
        Title of the plot.
    """
    # Create a DataFrame of feature importances
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feat_imp = feat_imp.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Take top n_top features
    feat_imp = feat_imp.head(n_top)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title(title)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_mrmr_feature_importances(feature_importance_df, output_file=None, figsize=(12, 10)):
    """
    Plot MRMR feature importances as a multi-panel figure.
    
    Parameters
    ----------
    feature_importance_df : pandas DataFrame
        DataFrame with feature importance information from MRMRFeatureSelector.
    output_file : str or Path, default=None
        If provided, save the plot to this file.
    figsize : tuple, default=(12, 10)
        Figure size.
    """
    plt.figure(figsize=figsize)
    
    # Plot relevance scores
    sorted_by_relevance = feature_importance_df.sort_values('Relevance', ascending=False)
    plt.subplot(2, 1, 1)
    sns.barplot(x='Relevance', y='Feature', data=sorted_by_relevance.head(20), palette='viridis')
    plt.title('Top 20 Features by Relevance Score')
    plt.tight_layout()
    
    # Plot MRMR scores for selected features
    selected_features = feature_importance_df[feature_importance_df['Selected']].sort_values('MRMR_Score', ascending=False)
    plt.subplot(2, 1, 2)
    sns.barplot(x='MRMR_Score', y='Feature', data=selected_features, palette='viridis')
    plt.title('Selected Features by MRMR Score')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_feature_selection_comparison(complete_metrics_df, figsize=(14, 8), output_file=None):
    """
    Create comparison plots for model performance with different feature selection methods.
    
    Parameters
    ----------
    complete_metrics_df : pandas DataFrame
        DataFrame with metrics for different feature selection methods.
    figsize : tuple, default=(14, 8)
        Figure size.
    output_file : str or Path, default=None
        If provided, save the plot to this file.
    """
    if 'FeatureSelectionMethod' not in complete_metrics_df.columns:
        print("No feature selection comparison available in metrics data.")
        return None
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_score', 'MCC']
    available_metrics = [m for m in metrics if m in complete_metrics_df.columns]
    
    plt.figure(figsize=figsize)
    
    for i, metric in enumerate(available_metrics):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x='FeatureSelectionMethod', y=metric, data=complete_metrics_df)
        plt.title(f'{metric} by Feature Selection Method')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_feature_count_comparison(complete_metrics_df, figsize=(12, 6), output_file=None):
    """
    Plot model performance vs number of features selected.
    
    Parameters
    ----------
    complete_metrics_df : pandas DataFrame
        DataFrame with metrics for different numbers of features selected.
    figsize : tuple, default=(12, 6)
        Figure size.
    output_file : str or Path, default=None
        If provided, save the plot to this file.
    """
    if 'FeaturesSelected' not in complete_metrics_df.columns:
        print("No feature count information available in metrics data.")
        return None
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_score', 'MCC']
    available_metrics = [m for m in metrics if m in complete_metrics_df.columns]
    
    plt.figure(figsize=figsize)
    
    for metric in available_metrics:
        plt.plot(complete_metrics_df['FeaturesSelected'], 
                complete_metrics_df[metric], 
                'o-', 
                label=metric)
    
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Performance Metric')
    plt.title('Model Performance vs Number of Features')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

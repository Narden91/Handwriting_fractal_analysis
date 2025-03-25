"""
Simple feature selection module using scikit-learn's SelectKBest.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import f_regression, mutual_info_regression
from rich.console import Console
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

console = Console()

# Dictionary of available scoring functions
SCORING_FUNCTIONS = {
    # Classification
    'f_classif': f_classif,  # ANOVA F-value 
    'chi2': chi2,  # Chi-squared stats (requires non-negative features)
    'mutual_info_classif': mutual_info_classif,  # Mutual information
    
    # Regression
    'f_regression': f_regression,  # F-value
    'mutual_info_regression': mutual_info_regression  # Mutual information
}

def select_k_best_features(X, y, k=10, score_func='f_classif', verbose=0):
    """
    Select top k features using SelectKBest with specified scoring function.
    
    Parameters
    ----------
    X : pandas DataFrame or numpy array
        Features
    y : pandas Series or numpy array
        Target variable
    k : int or float, default=10
        Number of features to select.
        If int, selects the k highest scoring features.
        If float between 0 and 1, selects k% of features.
    score_func : str or callable, default='f_classif'
        Function to score features. If string, must be one of:
        'f_classif', 'chi2', 'mutual_info_classif', 'f_regression', 'mutual_info_regression'
    verbose : int, default=0
        Controls verbosity of output
    
    Returns
    -------
    X_new : pandas DataFrame
        Transformed data with only selected features
    selector : SelectKBest
        Fitted selector that can be used to transform new data
    feature_scores : pandas DataFrame
        DataFrame with feature scores and selection status
    """
    # Convert inputs to pandas if they're not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Calculate k if it's a percentage
    if isinstance(k, float) and 0 < k < 1:
        k = max(1, int(k * X.shape[1]))
    else:
        k = min(k, X.shape[1])  # Can't select more features than we have
    
    # Handle NaN values
    if X.isna().any().any():
        if verbose > 0:
            console.print("[yellow]Warning: NaN values detected. Filling with mean/mode values.[/yellow]")
        
        # For numeric columns, fill NaNs with mean
        num_cols = X.select_dtypes(include=np.number).columns
        if not num_cols.empty:
            X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
        
        # For categorical columns, fill NaNs with mode
        cat_cols = X.select_dtypes(exclude=np.number).columns
        if not cat_cols.empty:
            for col in cat_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "UNKNOWN")
    
    # Get scoring function
    if isinstance(score_func, str):
        if score_func in SCORING_FUNCTIONS:
            score_func = SCORING_FUNCTIONS[score_func]
        else:
            valid_funcs = list(SCORING_FUNCTIONS.keys())
            console.print(f"[red]Error: Unknown scoring function '{score_func}'. Using 'f_classif' instead.[/red]")
            console.print(f"[yellow]Valid options are: {', '.join(valid_funcs)}[/yellow]")
            score_func = f_classif
    
    # Special handling for chi2 which requires non-negative features
    if score_func == chi2:
        if verbose > 0:
            console.print("[yellow]Chi2 requires non-negative features. Checking for negative values...[/yellow]")
        
        has_negative = False
        for col in X.select_dtypes(include=np.number).columns:
            if (X[col] < 0).any():
                has_negative = True
                if verbose > 0:
                    console.print(f"[yellow]Column {col} has negative values. Applying min-max scaling.[/yellow]")
                
                # Min-max scaling to make non-negative
                X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min() + 1e-8)
        
        if has_negative and verbose > 0:
            console.print("[yellow]Applied min-max scaling to make all features non-negative for chi2.[/yellow]")
    
    if verbose > 0:
        console.print(f"[bold cyan]Selecting top {k} features using {score_func.__name__}...[/bold cyan]")
    
    # Apply SelectKBest
    selector = SelectKBest(score_func=score_func, k=k)
    
    try:
        # Try with original data
        X_new = selector.fit_transform(X, y)
        
        # Convert back to DataFrame with selected feature names
        support = selector.get_support()
        selected_features = X.columns[support].tolist()
        
        X_new_df = pd.DataFrame(X_new, columns=selected_features, index=X.index)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_,
            'Selected': support
        }).sort_values('Score', ascending=False)
        
        if verbose > 0:
            console.print(f"[bold green]Successfully selected {len(selected_features)} features.[/bold green]")
            if verbose > 1:
                console.print(f"Selected features: {selected_features}")
        
        return X_new_df, selector, feature_scores
        
    except Exception as e:
        console.print(f"[bold red]Error in feature selection: {str(e)}[/bold red]")
        
        # Return original data if selection fails
        console.print("[yellow]Feature selection failed. Returning original data.[/yellow]")
        return X, None, pd.DataFrame({'Feature': X.columns, 'Score': np.ones(X.shape[1]), 'Selected': True})

def plot_feature_scores(feature_scores, output_file=None, figsize=(10, 8), top_n=20):
    """
    Plot feature scores from SelectKBest.
    
    Parameters
    ----------
    feature_scores : pandas DataFrame
        DataFrame with feature scores from select_k_best_features
    output_file : str or Path, default=None
        If provided, save plot to this file
    figsize : tuple, default=(10, 8)
        Figure size
    top_n : int, default=20
        Number of top features to display
    
    Returns
    -------
    fig : matplotlib Figure
        The figure object
    """
    plt.figure(figsize=figsize)
    
    # Sort by score
    sorted_scores = feature_scores.sort_values('Score', ascending=False).head(top_n)
    
    # Plot scores
    ax = sns.barplot(x='Score', y='Feature', data=sorted_scores, palette='viridis', 
                    hue='Selected', dodge=False)
    
    # Highlight selected features
    bars = ax.patches
    for i, bar in enumerate(bars):
        if sorted_scores.iloc[i]['Selected']:
            bar.set_facecolor('darkgreen')
        else:
            bar.set_facecolor('lightgrey')
    
    plt.title(f'Top {top_n} Feature Scores')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def save_feature_selection_results(feature_scores, selected_features, output_dir):
    """
    Save feature selection results to output directory.
    
    Parameters
    ----------
    feature_scores : pandas DataFrame
        DataFrame with feature scores
    selected_features : list
        List of selected feature names
    output_dir : str or Path
        Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature scores CSV
    feature_scores.to_csv(output_dir / "feature_scores.csv", index=False)
    
    # Save selected features text file
    with open(output_dir / "selected_features.txt", "w") as f:
        f.write("Selected Features:\n")
        for feature in selected_features:
            f.write(f"- {feature}\n")
    
    # Plot feature scores
    plot_feature_scores(feature_scores, output_file=output_dir / "feature_scores.png")
import hydra
from omegaconf import DictConfig
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
sys.dont_write_bytecode = True
from rich import print
from rich.console import Console
from rich.panel import Panel
from src.hp_tuning import run_hyperparameter_search
from src.feature_selection.select_best import select_k_best_features, save_feature_selection_results
from src.feature_selection.select_best import safe_apply_feature_selection


console = Console()


def ensure_dataset_alignment(X_train, X_test, verbose=True):
    """
    Ensure that training and test datasets are aligned and compatible.
    Performs checks and corrections on column presence, types, and order.
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features DataFrame
    X_test : pandas DataFrame
        Test features DataFrame
    verbose : bool
        Whether to print status messages
        
    Returns:
    --------
    X_train, X_test : pandas DataFrames
        Aligned and compatible DataFrames
    """
    if verbose:
        console.print("[bold]Ensuring dataset alignment before feature selection...[/bold]")
    
    # 1. Check for columns presence in both datasets
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols
    common_cols = list(train_cols.intersection(test_cols))
    
    if missing_in_test:
        if verbose:
            console.print(f"[yellow]Warning: {len(missing_in_test)} columns in train but not in test: {missing_in_test}[/yellow]")
        # Two options: 
        # 1. Add missing columns to test with NaN values (less strict)
        # 2. Remove these columns from train (more strict)
        # Using option 2 for consistency
        X_train = X_train.drop(columns=list(missing_in_test))
        if verbose:
            console.print(f"[yellow]Removed {len(missing_in_test)} columns from training set[/yellow]")
    
    if missing_in_train:
        if verbose:
            console.print(f"[yellow]Warning: {len(missing_in_train)} columns in test but not in train: {missing_in_train}[/yellow]")
        # Remove these columns from test
        X_test = X_test.drop(columns=list(missing_in_train))
        if verbose:
            console.print(f"[yellow]Removed {len(missing_in_train)} columns from test set[/yellow]")
    
    # 2. Ensure same column order
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # 3. Check for datatypes
    dtypes_train = X_train.dtypes
    dtypes_test = X_test.dtypes
    
    mismatched_types = {}
    for col in common_cols:
        if dtypes_train[col] != dtypes_test[col]:
            mismatched_types[col] = (dtypes_train[col], dtypes_test[col])
    
    if mismatched_types:
        if verbose:
            console.print(f"[yellow]Warning: {len(mismatched_types)} columns have mismatched types:[/yellow]")
            for col, (train_type, test_type) in mismatched_types.items():
                console.print(f"  - {col}: train={train_type}, test={test_type}")
        
        # Fix type mismatches by converting to common type
        for col, (train_type, test_type) in mismatched_types.items():
            # If one is numeric and other is categorical, prefer numeric
            if pd.api.types.is_numeric_dtype(train_type) and not pd.api.types.is_numeric_dtype(test_type):
                if verbose:
                    console.print(f"  Converting {col} in test to {train_type}")
                X_test[col] = X_test[col].astype(train_type)
            elif pd.api.types.is_numeric_dtype(test_type) and not pd.api.types.is_numeric_dtype(train_type):
                if verbose:
                    console.print(f"  Converting {col} in train to {test_type}")
                X_train[col] = X_train[col].astype(test_type)
            else:
                # Default: make test match train
                if verbose:
                    console.print(f"  Converting {col} in test to match train ({train_type})")
                try:
                    X_test[col] = X_test[col].astype(train_type)
                except:
                    # If conversion fails, use a more generic type (object)
                    X_train[col] = X_train[col].astype('object')
                    X_test[col] = X_test[col].astype('object')
    
    # 4. Check for NaN values
    train_nan_cols = X_train.columns[X_train.isna().any()].tolist()
    test_nan_cols = X_test.columns[X_test.isna().any()].tolist()
    
    if train_nan_cols or test_nan_cols:
        if verbose:
            console.print(f"[yellow]Warning: NaN values found in datasets[/yellow]")
            if train_nan_cols:
                console.print(f"  - Training set: {len(train_nan_cols)} columns with NaNs")
            if test_nan_cols:
                console.print(f"  - Test set: {len(test_nan_cols)} columns with NaNs")
        
        # Impute NaN values to ensure consistency
        for col in set(train_nan_cols).union(set(test_nan_cols)):
            if pd.api.types.is_numeric_dtype(X_train[col]):
                # For numeric columns, use median
                col_median = X_train[col].median()
                if pd.isna(col_median):  # If median itself is NaN
                    col_median = 0
                X_train[col] = X_train[col].fillna(col_median)
                X_test[col] = X_test[col].fillna(col_median)
                if verbose:
                    console.print(f"  - Imputed NaNs in {col} with median: {col_median}")
            else:
                # For categorical columns, use mode
                col_mode = X_train[col].mode()[0] if not X_train[col].mode().empty else "UNKNOWN"
                X_train[col] = X_train[col].fillna(col_mode)
                X_test[col] = X_test[col].fillna(col_mode)
                if verbose:
                    console.print(f"  - Imputed NaNs in {col} with mode: {col_mode}")
    
    # 5. Check index alignment (not usually a problem for feature selection but good practice)
    if not X_train.index.equals(X_test.index) and len(X_train) == len(X_test):
        if verbose:
            console.print("[yellow]Warning: Indices are not aligned. Resetting indices.[/yellow]")
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
    
    if verbose:
        console.print(f"[green]Datasets aligned: {X_train.shape[1]} features in each[/green]")
    
    return X_train, X_test


def process_task(task_path, db_info, config, run_seed, run_idx, task_idx):
    """Process a single task with the provided seed."""
    console.print(f"[bold cyan]Processing Task {task_idx+1}: {task_path.name} (Run {run_idx+1}, Seed {run_seed})[/bold cyan]")

    # Load task data
    task_df = pd.read_csv(task_path)
    task_df = task_df.merge(db_info, on="Id", how="left")

    class_col = task_df.pop("Class")
    task_df["Class"] = class_col

    if config.settings.verbose > 0:
        console.print(task_df.head())

    X_train, X_test, y_train, y_test = train_test_split(
        task_df.drop("Class", axis=1), task_df["Class"],
        test_size=0.2, random_state=run_seed, stratify=task_df["Class"])

    X_train = X_train.drop("Id", axis=1)
    X_test = X_test.drop("Id", axis=1)

    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        console.print("[bold red]NaN values found before scaling. Imputing Values[/bold red]")
        X_train.fillna(X_train.median(), inplace=True)
        X_test.fillna(X_test.median(), inplace=True)

    scaler = StandardScaler() if config.preprocessing.scaler == "Standard" else RobustScaler()

    # Extract binary features (don't scale these)
    X_train_no_scale = X_train[["Work", "Sex"]]
    X_test_no_scale = X_test[["Work", "Sex"]]

    # Remove binary features from the datasets to be scaled
    X_train_to_scale = X_train.drop(["Work", "Sex"], axis=1)
    X_test_to_scale = X_test.drop(["Work", "Sex"], axis=1)

    # Scale the non-binary features
    scaled_columns = X_train_to_scale.columns
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_to_scale),
                                columns=scaled_columns,
                                index=X_train_to_scale.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_to_scale),
                                columns=scaled_columns,
                                index=X_test_to_scale.index)

    X_train = pd.concat([X_train_scaled.reset_index(drop=True),
                        X_train_no_scale.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test_scaled.reset_index(drop=True),
                        X_test_no_scale.reset_index(drop=True)], axis=1)
    
    
    # --- FEATURE SELECTION ---
    if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
        console.print("[bold]Starting feature selection...[/bold]") if config.settings.verbose > 0 else None

        # Ensure datasets are aligned before feature selection
        X_train, X_test = ensure_dataset_alignment(X_train, X_test, verbose=config.settings.verbose > 0)
        
        feature_selection_method = config.feature_selection.get('method', 'selectkbest')

        fs_output_dir = None
        if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
            base_dir = Path(config.hyperparameter_tuning.output_dir)
            fs_dir = base_dir / config.hyperparameter_tuning.models[0] / "feature_selection" 
            run_dir = fs_dir / f"run_{run_idx+1}"
            task_dir = run_dir / f"task_{task_idx+1}"
            task_dir.mkdir(parents=True, exist_ok=True)
            fs_output_dir = task_dir

        try:
            if feature_selection_method == 'selectkbest':
                k = config.feature_selection.get('k', 10)
                score_func = config.feature_selection.get('score_func', 'f_classif')
                verbose_level = 2 if config.settings.verbose > 0 else 0

                # Save original shapes for debugging
                orig_train_shape = X_train.shape
                orig_test_shape = X_test.shape
                
                # Try feature selection with error handling
                try:
                    X_train_new, selector, feature_scores = select_k_best_features(
                        X_train, y_train,
                        k=k,
                        score_func=score_func,
                        verbose=verbose_level,
                        random_state=run_seed  
                    )
                    
                    if selector:
                        # Log feature selection results
                        n_selected = sum(selector.get_support())
                        console.print(f"[green]Selected {n_selected} out of {len(selector.get_support())} features[/green]")
                        
                        # Apply selection to both datasets
                        X_train, X_test = safe_apply_feature_selection(
                            selector, X_train, X_test, verbose=verbose_level
                        )
                        
                        # Verify the output shapes
                        console.print(f"Original shapes: Train {orig_train_shape}, Test {orig_test_shape}")
                        console.print(f"New shapes: Train {X_train.shape}, Test {X_test.shape}")
                        
                        if fs_output_dir is not None and config.feature_selection.get('save_plots', True):
                            selected_features = X_train.columns.tolist()
                            save_feature_selection_results(
                                feature_scores,
                                selected_features,
                                fs_output_dir
                            )
                    else:
                        console.print("[yellow]Feature selection returned all features.[/yellow]")
                except Exception as e:
                    console.print(f"[bold red]Error during feature selection: {str(e)}[/bold red]")
                    # Get traceback for more detailed error info
                    import traceback
                    error_traceback = traceback.format_exc()
                    console.print(f"[dim]{error_traceback}[/dim]")
                    console.print("[yellow]Continuing with all features...[/yellow]")

                    if fs_output_dir is not None:
                        with open(fs_output_dir / "error_log.txt", "w") as f:
                            f.write(f"Feature selection error: {str(e)}\n")
                            f.write(error_traceback)
            else:
                console.print(f"[yellow]Unknown feature selection method: {feature_selection_method}. Skipping feature selection.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Unexpected error in feature selection section: {str(e)}[/bold red]")
            # Continue with the original datasets
            console.print("[yellow]Continuing with all features...[/yellow]")

    # --- MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION ---
    console.print("[bold]Starting hyperparameter optimization...[/bold]") if config.settings.verbose > 0 else None

    # Format: results/model_type/run_N/task_M/
    task_output_dirs = {}
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        base_dir = Path(config.hyperparameter_tuning.output_dir)
        for model in config.hyperparameter_tuning.models:
            model_dir = base_dir / model
            run_dir = model_dir / f"run_{run_idx+1}"
            task_dir = run_dir / f"task_{task_idx+1}"
            task_dir.mkdir(parents=True, exist_ok=True)
            task_output_dirs[model] = task_dir

    # Get hyperparameter tuning parameters from config
    models_to_optimize = config.hyperparameter_tuning.models
    n_trials = config.hyperparameter_tuning.n_trials
    metric = config.hyperparameter_tuning.metric
    cv = config.hyperparameter_tuning.cv

    # Run hyperparameter optimization with train/test data
    results = run_hyperparameter_search(
        models_to_optimize=models_to_optimize,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_trials=n_trials,
        cv=cv,
        metric=metric,
        output_dir=task_output_dirs,  
        run_seed=run_seed 
    )

    best_model_type = max(results, key=lambda k: results[k]['test_accuracy'])
    best_model_accuracy = results[best_model_type]['test_accuracy']

    console.print(f"[bold green]Best model for Task {task_idx+1} (Run {run_idx+1}): {best_model_type} with accuracy {best_model_accuracy:.4f}[/bold green]") if config.settings.verbose > 0 else None

    return results
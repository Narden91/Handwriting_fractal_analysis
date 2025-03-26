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


console = Console()


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
        console.print("[bold]Starting feature selection...[/bold]")

        feature_selection_method = config.feature_selection.get('method', 'selectkbest')

        fs_output_dir = None
        if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
            base_dir = Path(config.hyperparameter_tuning.output_dir)
            fs_dir = base_dir / config.hyperparameter_tuning.models[0] / "feature_selection" 
            run_dir = fs_dir / f"run_{run_idx+1}"
            task_dir = run_dir / f"task_{task_idx+1}"
            task_dir.mkdir(parents=True, exist_ok=True)
            fs_output_dir = task_dir

        if feature_selection_method == 'selectkbest':
            k = config.feature_selection.get('k', 10)
            score_func = config.feature_selection.get('score_func', 'f_classif')
            verbose_level = 2 if config.settings.verbose > 0 else 0

            try:
                X_train, selector, feature_scores = select_k_best_features(
                    X_train, y_train,
                    k=k,
                    score_func=score_func,
                    verbose=verbose_level,
                    random_state=run_seed  
                )

                if selector:
                    selected_features = X_train.columns.tolist()
                    X_test = X_test[selected_features]

                    if fs_output_dir is not None and config.feature_selection.get('save_plots', True):
                        save_feature_selection_results(
                            feature_scores,
                            selected_features,
                            fs_output_dir
                        )

                    console.print(f"[bold green]Feature selection completed. Selected {len(selected_features)} features.[/bold green]")
                else:
                    console.print("[yellow]Feature selection returned all features.[/yellow]")

            except Exception as e:
                console.print(f"[bold red]Error during feature selection: {str(e)}[/bold red]")
                console.print("[yellow]Continuing with all features...[/yellow]")

                if fs_output_dir is not None:
                    with open(fs_output_dir / "error_log.txt", "w") as f:
                        f.write(f"Feature selection error: {str(e)}\n")

        else:
            console.print(f"[yellow]Unknown feature selection method: {feature_selection_method}. Skipping feature selection.[/yellow]")

    # --- MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION ---
    console.print("[bold]Starting hyperparameter optimization...[/bold]")

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

    console.print(f"[bold green]Best model for Task {task_idx+1} (Run {run_idx+1}): {best_model_type} with accuracy {best_model_accuracy:.4f}[/bold green]")

    return results
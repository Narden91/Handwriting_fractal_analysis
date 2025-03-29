"""
Cross-task handwriting analysis module.
Performs ML classification across multiple tasks with subject-based train/test splits.
"""
import random
import hydra
from omegaconf import DictConfig
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import os

sys.dont_write_bytecode = True
from rich import print
from rich.console import Console
from rich.panel import Panel
from src.hp_tuning import run_hyperparameter_search
from src.feature_selection import select_k_best_features, save_feature_selection_results
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler

console = Console()


def set_global_seeds(seed):
    """Set all common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def cross_task_analysis(data_df, id_column, task_column, class_column, config, run_seed, run_idx):
    """
    Perform cross-task analysis with subject-wise train/test split.
    
    Args:
        data_df: DataFrame with features and labels
        id_column: Column name for subject IDs
        task_column: Column name for task identifiers
        class_column: Column name for class labels
        config: Configuration from Hydra
        run_seed: Random seed for this run
        run_idx: Index of the current run
        
    Returns:
        Dictionary with results
    """
    console.print(f"[bold cyan]Processing Cross-Task Analysis (Run {run_idx+1}, Seed {run_seed})[/bold cyan]")
    
    # Extract subject IDs for subject-wise split
    subject_ids = data_df[id_column].unique()
    console.print(f"Found {len(subject_ids)} unique subjects across {data_df[task_column].nunique()} tasks")
    
    # Perform subject-wise split using GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=run_seed)
    train_idx, test_idx = next(splitter.split(data_df, groups=data_df[id_column]))
    
    train_df = data_df.iloc[train_idx]
    test_df = data_df.iloc[test_idx]
    
    # Verify subject separation between train and test
    train_subjects = set(train_df[id_column].unique())
    test_subjects = set(test_df[id_column].unique())
    assert len(train_subjects.intersection(test_subjects)) == 0, "Subject overlap between train and test sets!"
    
    console.print(f"Train set: {len(train_df)} samples from {len(train_subjects)} subjects")
    console.print(f"Test set: {len(test_df)} samples from {len(test_subjects)} subjects")
    
    # Extract task distribution in splits (for detailed logging)
    if config.settings.verbose > 0:
        train_tasks = train_df[task_column].value_counts().to_dict()
        test_tasks = test_df[task_column].value_counts().to_dict()
        console.print(f"Task distribution in train set: {train_tasks}")
        console.print(f"Task distribution in test set: {test_tasks}")
    
    # Separate features and labels, removing only ID and class columns
    # We'll handle the task column specially as categorical data
    feature_cols = [col for col in data_df.columns if col not in [id_column, class_column]]
    X_train = train_df[feature_cols].copy()
    y_train = train_df[class_column]
    X_test = test_df[feature_cols].copy()
    y_test = test_df[class_column]
    
    # Create one-hot encoding for the Task column
    console.print(f"One-hot encoding the {task_column} column")
    
    # Remove the original task column from features (we'll replace with encoded version)
    X_train_task = X_train[task_column]
    X_test_task = X_test[task_column]
    X_train = X_train.drop(task_column, axis=1)
    X_test = X_test.drop(task_column, axis=1)
    
    # Create one-hot encoding
    task_dummies_train = pd.get_dummies(X_train_task, prefix='task')
    task_dummies_test = pd.get_dummies(X_test_task, prefix='task')
    
    # Handle potential missing categories in test set
    for col in task_dummies_train.columns:
        if col not in task_dummies_test.columns:
            task_dummies_test[col] = 0
    
    # Ensure test dummies have same columns as train (and in same order)
    task_dummies_test = task_dummies_test[task_dummies_train.columns]
    
    # Now add the encoded tasks back to features
    X_train = pd.concat([X_train, task_dummies_train], axis=1)
    X_test = pd.concat([X_test, task_dummies_test], axis=1)
    
    if config.settings.verbose > 0:
        console.print(f"After one-hot encoding: {X_train.shape[1]} features")
        console.print(f"Task features: {list(task_dummies_train.columns)}")
    
    # Scaling features - follow the same approach as in the original code
    scaler = StandardScaler() if config.preprocessing.scaler == "Standard" else RobustScaler()
    
    # Extract binary features if they exist (don't scale these)
    binary_columns = []
    if all(col in X_train.columns for col in ["Work", "Sex"]):
        binary_columns = ["Work", "Sex"]
        X_train_no_scale = X_train[binary_columns]
        X_test_no_scale = X_test[binary_columns]
        
        X_train_to_scale = X_train.drop(binary_columns, axis=1)
        X_test_to_scale = X_test.drop(binary_columns, axis=1)
        
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
    else:
        # Scale all features
        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                            columns=X_train.columns,
                            index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test),
                            columns=X_test.columns,
                            index=X_test.index)
    
    # Feature selection if enabled
    if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
        console.print("[bold]Starting feature selection...[/bold]") if config.settings.verbose > 0 else None
        
        feature_selection_method = config.feature_selection.get('method', 'selectkbest')
        
        fs_output_dir = None
        if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
            base_dir = Path(config.hyperparameter_tuning.output_dir)
            fs_dir = base_dir / "cross_task" / config.hyperparameter_tuning.models[0] / "feature_selection" 
            run_dir = fs_dir / f"run_{run_idx+1}"
            run_dir.mkdir(parents=True, exist_ok=True)
            fs_output_dir = run_dir
        
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
    
    # Model training with hyperparameter optimization
    console.print("[bold]Starting hyperparameter optimization...[/bold]") if config.settings.verbose > 0 else None
    
    # Format: results/cross_task/model_type/run_N/
    task_output_dirs = {}
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        base_dir = Path(config.hyperparameter_tuning.output_dir)
        cross_task_dir = base_dir / "cross_task"
        for model in config.hyperparameter_tuning.models:
            model_dir = cross_task_dir / model
            run_dir = model_dir / f"run_{run_idx+1}"
            run_dir.mkdir(parents=True, exist_ok=True)
            task_output_dirs[model] = run_dir
    
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
    
    console.print(f"[bold green]Best model for Cross-Task Analysis (Run {run_idx+1}): {best_model_type} with accuracy {best_model_accuracy:.4f}[/bold green]") if config.settings.verbose > 0 else None
    
    return results


@hydra.main(config_path="./config", config_name="ml_config_cross_task", version_base="1.2")
def main(config: DictConfig):
    """Main function to run the cross-task ML pipeline."""
    start_time = time.time()
    console.print(Panel("[bold cyan]🔍 Cross-Task ML Fractal Handwriting Analysis[/bold cyan]",
                      title="Starting Analysis", expand=False))
    
    if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
        fs_method = config.feature_selection.method
        k_value = config.feature_selection.k
        score_func = config.feature_selection.get('score_func', 'f_classif')
        console.print(Panel(f"[bold magenta]Feature Selection: {fs_method.upper()} (k={k_value}, scoring={score_func})[/bold magenta]",
                          expand=False))
    
    verbose = config.settings.verbose
    debug = config.settings.debug
    n_runs = config.settings.n_runs
    base_seed = config.settings.base_seed
    set_global_seeds(base_seed)
    
    # Load the concatenated data file
    data_path = Path(config.data.path)
    csv_file = data_path / config.data.concat_file
    
    console.print(f"Loading data from {csv_file}")
    if not csv_file.exists():
        console.print(f"[bold red]Error: File not found: {csv_file}[/bold red]")
        return {}
    
    try:
        data_df = pd.read_csv(csv_file)
        console.print(f"Loaded dataset with {data_df.shape[0]} rows and {data_df.shape[1]} columns")
        
        if verbose > 0:
            console.print("Data sample:")
            console.print(data_df.head())
            
            console.print("Column information:")
            for col in data_df.columns:
                console.print(f"- {col}: {data_df[col].dtype}")
            
            # Show distribution of classes and tasks
            console.print(f"Class distribution: {data_df[config.data.class_column].value_counts().to_dict()}")
            console.print(f"Task distribution: {data_df[config.data.task_column].value_counts().to_dict()}")
            console.print(f"Number of unique subjects: {data_df[config.data.id_column].nunique()}")
            
            # Show samples per subject distribution
            samples_per_subj = data_df.groupby(config.data.id_column).size()
            console.print(f"Samples per subject - Min: {samples_per_subj.min()}, Max: {samples_per_subj.max()}, Avg: {samples_per_subj.mean():.2f}")
            
    except Exception as e:
        console.print(f"[bold red]Error loading data: {str(e)}[/bold red]")
        return {}
    
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        main_output_dir = Path(config.hyperparameter_tuning.output_dir)        
        main_output_dir.mkdir(parents=True, exist_ok=True)
    
    overall_results = {}
    
    for run_idx in range(n_runs):
        run_seed = base_seed + run_idx
        console.print(Panel(f"[bold]Starting Run {run_idx+1}/{n_runs} with Seed {run_seed}[/bold]",
                          expand=False))
        
        run_results = cross_task_analysis(
            data_df=data_df,
            id_column=config.data.id_column,
            task_column=config.data.task_column,
            class_column=config.data.class_column,
            config=config,
            run_seed=run_seed,
            run_idx=run_idx
        )
        
        overall_results[f"run_{run_idx+1}"] = run_results
        
        if debug and run_idx == 0:
            console.print("[bold yellow]Debug mode: stopping after first run[/bold yellow]")
            break
    
    # Calculate and save overall performance metrics
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        model_metrics = {}
        
        for model_type in config.hyperparameter_tuning.models:
            model_metrics[model_type] = []
        
        # Loop through all runs to collect metrics
        for run_idx in range(min(n_runs, 1 if debug else float('inf'))):
            run_seed = base_seed + run_idx
            
            # For each model type, read the metrics.json file
            for model_type in config.hyperparameter_tuning.models:
                if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
                    base_dir = Path(config.hyperparameter_tuning.output_dir)
                    cross_task_dir = base_dir / "cross_task"
                    model_dir = cross_task_dir / model_type
                    run_dir = model_dir / f"run_{run_idx+1}"
                    metrics_file = run_dir / "metrics.json"
                    
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        metrics['Run'] = run_idx + 1
                        metrics['Seed'] = run_seed
                        
                        if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
                            metrics['FeatureSelectionMethod'] = config.feature_selection.method
                            metrics['K'] = config.feature_selection.k
                            metrics['ScoreFunc'] = config.feature_selection.get('score_func', 'f_classif')
                        
                        model_metrics[model_type].append(metrics)
        
        # Create performance summary for each model
        for model_type, metrics_list in model_metrics.items():
            if metrics_list:
                model_dir = Path(config.hyperparameter_tuning.output_dir) / "cross_task" / model_type
                df = pd.DataFrame(metrics_list)
                
                column_mapping = {
                    'accuracy': 'Accuracy',
                    'precision': 'Precision',
                    'recall': 'Recall',
                    'specificity': 'Specificity',
                    'mcc': 'MCC',
                    'f1_score': 'F1_score'
                }
                df.rename(columns=column_mapping, inplace=True)
                
                # Create summary of averages per run
                metrics_to_aggregate = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'MCC', 'F1_score']
                available_metrics = [col for col in metrics_to_aggregate if col in df.columns]
                agg_dict = {metric: ['mean', 'std'] for metric in available_metrics}
                
                run_summary = df.groupby('Run').agg(agg_dict).reset_index()
                
                # Flatten multi-level column names
                run_summary.columns = ['_'.join(col).strip('_') for col in run_summary.columns.values]
                run_summary.to_csv(model_dir / "run_performance_summary.csv", index=False)
                
                if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
                    df.to_csv(model_dir / "complete_metrics.csv", index=False)
    
    end_time = time.time()
    execution_time = end_time - start_time
    console.print(f"[bold]Total execution time: {execution_time:.2f} seconds[/bold]")
    
    return overall_results


if __name__=="__main__":
    main()
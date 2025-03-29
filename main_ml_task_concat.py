"""
Task concatenation analysis module for handwriting features.
Performs ML classification by concatenating features from multiple tasks.
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
import matplotlib.pyplot as plt
import seaborn as sns

sys.dont_write_bytecode = True
from rich import print
from rich.console import Console
from rich.panel import Panel
from src.hp_tuning import run_hyperparameter_search
from src.feature_selection import select_k_best_features, save_feature_selection_results
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

console = Console()


def set_global_seeds(seed):
    """Set all common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def concatenate_task_features(data_path, verbose=0):
    """
    Concatenate features from all task files in the data_path.
    Each task's features will be prefixed with 'Task_N_' to avoid column conflicts.
    
    Args:
        data_path: Path containing TASK_*.csv files
        verbose: Verbosity level for logging
        
    Returns:
        DataFrame with concatenated features, with ID as first column and Class as last column
    """
    task_files = sorted([f for f in data_path.glob("TASK_*.csv")])
    if verbose > 0:
        console.print(f"Found {len(task_files)} task files for concatenation")
    
    if not task_files:
        console.print("[bold red]No task files found for concatenation[/bold red]")
        return None
    
    # First, identify all unique IDs across all task files
    all_ids = set()
    for file_path in task_files:
        df = pd.read_csv(file_path)
        all_ids.update(df["Id"].unique())
    
    all_ids = sorted(list(all_ids))
    if verbose > 0:
        console.print(f"Found {len(all_ids)} unique IDs across all tasks")
    
    # Track class labels in case some subjects don't have all tasks
    all_classes = {}
    
    # Initialize empty DataFrame for concatenation
    concat_feat_df = None
    id_col = None
    class_col = None
    
    # Process each task file
    for idx, file_path in enumerate(task_files):
        # Load the csv file into a dataframe
        file = file_path.name
        if verbose > 0:
            console.print(f"Processing file: {file}")
        
        df_new = pd.read_csv(file_path)
        
        # For the first file, save Id and Class columns
        if idx == 0:
            id_col = df_new["Id"].copy()
            class_col = df_new["Class"].copy()
        
        # Track class labels for each ID
        for _, row in df_new[["Id", "Class"]].iterrows():
            all_classes[row["Id"]] = row["Class"]
        
        # Remove Id and Class columns from the current dataframe
        if "Id" in df_new.columns:
            df_new.drop("Id", axis=1, inplace=True)
        if "Class" in df_new.columns:
            df_new.drop("Class", axis=1, inplace=True)
        
        # Add prefix to column names to avoid duplicates
        prefix = f"Task_{idx+1}_"
        df_new = df_new.add_prefix(prefix)
        
        # Concatenate with the main dataframe
        if idx == 0:
            concat_feat_df = df_new
        else:
            concat_feat_df = pd.concat([concat_feat_df, df_new], axis=1)
    
    # Add Id as the first column and Class as the last column
    concat_feat_df.insert(0, "Id", id_col)
    concat_feat_df["Class"] = class_col
    
    # Verify that the dataframe has the correct structure
    if verbose > 0:
        console.print(f"Concatenated dataframe has {concat_feat_df.shape[0]} rows and {concat_feat_df.shape[1]} columns")
        console.print(f"Class distribution: {concat_feat_df['Class'].value_counts().to_dict()}")
    
    return concat_feat_df


def optimize_dataframe_memory(df, verbose=0):
    """
    Optimize memory usage of the dataframe by downcasting numeric types.
    
    Args:
        df: pandas DataFrame to optimize
        verbose: Verbosity level for logging
        
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose > 0:
        console.print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose > 0:
        console.print(f"Memory usage after optimization is {end_mem:.2f} MB")
        console.print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df


def save_concatenated_dataset(concat_df, output_dir, run_idx=0):
    """
    Save the concatenated dataset for future reference.
    
    Args:
        concat_df: DataFrame with concatenated features
        output_dir: Directory to save the dataset
        run_idx: Run index (default: 0)
    """
    # Create the output directory if it doesn't exist
    save_dir = Path(output_dir) / "datasets"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
    output_file = save_dir / f"concatenated_features_run_{run_idx+1}.csv"
    concat_df.to_csv(output_file, index=False)
    
    # Also save a version with only ID and Class for reference
    id_class_df = concat_df[["Id", "Class"]].copy()
    id_class_file = save_dir / f"id_class_mapping_run_{run_idx+1}.csv"
    id_class_df.to_csv(id_class_file, index=False)
    
    return output_file


def generate_performance_visualizations(results_dir, model_metrics, run_idx=0):
    """
    Generate performance visualizations for the task concatenation analysis.
    
    Args:
        results_dir: Directory to save visualizations
        model_metrics: Dictionary with model metrics
        run_idx: Run index to visualize (default: 0 for first run)
    """
    # Set up the visualization directory
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparative model performance plot
    plt.figure(figsize=(12, 8))
    
    # Extract metrics for all models
    models = []
    accuracies = []
    f1_scores = []
    
    for model_type, metrics_list in model_metrics.items():
        if metrics_list:
            for metrics in metrics_list:
                if metrics['Run'] == run_idx + 1:
                    models.append(model_type)
                    accuracies.append(metrics.get('accuracy', 0))
                    f1_scores.append(metrics.get('f1_score', 0))
    
    # Create a dataframe for plotting
    df_plot = pd.DataFrame({
        'Model': models,
        'accuracy': accuracies,
        'F1 Score': f1_scores
    })
    
    # Melt the dataframe for easier plotting
    df_melt = pd.melt(df_plot, id_vars=['Model'], var_name='Metric', value_name='Value')
    
    # Debug info
    console.print(f"[bold]Single run visualization data (Run {run_idx+1}):[/bold]")
    console.print(f"Models: {models}")
    console.print(f"Accuracies: {accuracies}")
    console.print(f"F1 Scores: {f1_scores}")
    
    if len(df_melt) > 0:
        # Create the plot
        sns.barplot(x='Model', y='Value', hue='Metric', data=df_melt)
        plt.title(f'Model Performance Comparison (Run {run_idx+1})')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        ax = plt.gca()
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'bottom',
                        xytext = (0, 5), textcoords = 'offset points')
    else:
        # Create empty plot with message
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f"No performance data available for Run {run_idx+1}", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=16)
        plt.title(f'Model Performance Comparison (Run {run_idx+1})')
        plt.xlabel('Model')
        plt.ylabel('Value')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(viz_dir / f"model_comparison_run_{run_idx+1}.png", dpi=300)
    plt.close()
    
    # If feature importances are available, plot them for each model
    for model_type in model_metrics.keys():
        model_dir = results_dir / model_type / f"run_{run_idx+1}"
        fi_file = model_dir / "feature_importances.csv"
        
        if fi_file.exists():
            # Load feature importances
            try:
                fi_df = pd.read_csv(fi_file)
                
                # Plot top 20 features
                plt.figure(figsize=(12, 10))
                sns.barplot(x='Importance', y='Feature', data=fi_df.head(20))
                plt.title(f'Top 20 Feature Importances for {model_type.upper()} (Run {run_idx+1})')
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(viz_dir / f"{model_type}_feature_importances_run_{run_idx+1}.png", dpi=300)
                plt.close()
            except Exception as e:
                console.print(f"[yellow]Error creating feature importance plot for {model_type}: {str(e)}[/yellow]")


def task_concatenation_analysis(config, run_seed, run_idx):
    """
    Perform machine learning analysis on concatenated task features.
    
    Args:
        config: Configuration from Hydra
        run_seed: Random seed for this run
        run_idx: Index of the current run
        
    Returns:
        Dictionary with results
    """
    console.print(f"[bold cyan]Processing Task Concatenation Analysis (Run {run_idx+1}, Seed {run_seed})[/bold cyan]")
    
    # Extract paths
    data_path = Path(config.data.path)
    features_path = data_path / Path(config.data.feat_folder)
    
    # Load database info
    db_info = pd.read_csv(features_path / Path(config.data.db_info))
    rename_dict = {
        "id": "Id",
        "eta": "Age",
        "professione": "Work",
        "scolarizzazione": "Education",
        "sesso": "Sex"}
    db_info.rename(columns=rename_dict, inplace=True)
    
    # Concatenate features from all task files
    concat_df = concatenate_task_features(features_path, verbose=config.settings.verbose)
    
    if concat_df is None:
        console.print("[bold red]Failed to concatenate task features[/bold red]")
        return {}
    
    # Optimize memory usage
    concat_df = optimize_dataframe_memory(concat_df, verbose=config.settings.verbose)
    
    # Skip saving concatenated dataset files
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        save_dir = Path(config.hyperparameter_tuning.output_dir) / "task_concat"
    
    # Merge with database info
    full_df = concat_df.merge(db_info, on="Id", how="left")
    
    # Move Class to end (avoid DataFrame fragmentation)
    class_col = full_df["Class"].copy()
    # Drop it from the original
    full_df = full_df.drop("Class", axis=1)
    # Create new dataframe with all columns including Class at the end
    full_df = pd.concat([full_df, class_col.rename("Class")], axis=1)
    
    if config.settings.verbose > 0:
        console.print(f"Concatenated dataframe shape: {full_df.shape}")
        console.print(f"Number of features: {full_df.shape[1]-2}")  # -2 for Id and Class
        console.print(f"Number of samples: {full_df.shape[0]}")
        console.print(f"Class distribution: {full_df['Class'].value_counts().to_dict()}")
    
    # Split data into features and target
    X = full_df.drop(["Class"], axis=1)
    y = full_df["Class"]
    
    # Split into train and test sets (preserving Id column for reference)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=run_seed, stratify=y
    )
    
    # Save IDs for reference
    train_ids = X_train["Id"].copy()
    test_ids = X_test["Id"].copy()
    
    # Remove Id from features
    X_train = X_train.drop("Id", axis=1)
    X_test = X_test.drop("Id", axis=1)
    
    # Handle NaN values
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        console.print("[yellow]NaN values found before scaling. Imputing with median.[/yellow]")
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_test.median())
    
    # Scale features
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
        
        X_train = pd.concat([X_train_scaled, X_train_no_scale], axis=1)
        X_test = pd.concat([X_test_scaled, X_test_no_scale], axis=1)
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
            fs_dir = base_dir / "task_concat" / config.hyperparameter_tuning.models[0] / "feature_selection" 
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
    
    # Format: results/task_concat/model_type/run_N/
    task_output_dirs = {}
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        base_dir = Path(config.hyperparameter_tuning.output_dir)
        task_concat_dir = base_dir / "task_concat"
        for model in config.hyperparameter_tuning.models:
            model_dir = task_concat_dir / model
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
    
    console.print(f"[bold green]Best model for Task Concatenation Analysis (Run {run_idx+1}): {best_model_type} with accuracy {best_model_accuracy:.4f}[/bold green]") if config.settings.verbose > 0 else None
    
    # Skip saving ID splits
    
    return results


@hydra.main(config_path="./config", config_name="ml_config_task_concat", version_base="1.2")
def main(config: DictConfig):
    """Main function to run the task concatenation ML pipeline."""
    start_time = time.time()
    console.print(Panel("[bold cyan]🔍 Task Concatenation ML Analysis[/bold cyan]",
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
    
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        main_output_dir = Path(config.hyperparameter_tuning.output_dir) / "task_concat"        
        main_output_dir.mkdir(parents=True, exist_ok=True)
    
    overall_results = {}
    model_metrics = {model: [] for model in config.hyperparameter_tuning.models}
    
    for run_idx in range(n_runs):
        run_seed = base_seed + run_idx
        console.print(Panel(f"[bold]Starting Run {run_idx+1}/{n_runs} with Seed {run_seed}[/bold]",
                          expand=False))
        
        run_results = task_concatenation_analysis(config, run_seed, run_idx)
        
        overall_results[f"run_{run_idx+1}"] = run_results
        
                    # Collect metrics for this run
        if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
            for model_type in config.hyperparameter_tuning.models:
                metrics_file = Path(config.hyperparameter_tuning.output_dir) / "task_concat" / model_type / f"run_{run_idx+1}" / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        metrics['Run'] = run_idx + 1
                        metrics['Seed'] = run_seed
                        
                        if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
                            metrics['FeatureSelectionMethod'] = config.feature_selection.method
                            metrics['K'] = config.feature_selection.k
                            metrics['ScoreFunc'] = config.feature_selection.get('score_func', 'f_classif')
                        
                        model_metrics[model_type].append(metrics)
                        console.print(f"[green]Added metrics for {model_type} (Run {run_idx+1})[/green]")
                    except Exception as e:
                        console.print(f"[bold red]Error loading metrics for {model_type}: {str(e)}[/bold red]")
                else:
                    console.print(f"[yellow]Metrics file not found for {model_type} (Run {run_idx+1}): {metrics_file}[/yellow]")
        
        if debug and run_idx == 0:
            console.print("[bold yellow]Debug mode: stopping after first run[/bold yellow]")
            break
    
    # Calculate and save overall performance metrics
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        # Create performance summary for each model
        for model_type, metrics_list in model_metrics.items():
            if metrics_list:
                model_dir = Path(config.hyperparameter_tuning.output_dir) / "task_concat" / model_type
                df = pd.DataFrame(metrics_list)
                
                column_mapping = {
                    'accuracy': 'accuracy',
                    'precision': 'Precision',
                    'recall': 'Recall',
                    'specificity': 'Specificity',
                    'mcc': 'MCC',
                    'f1_score': 'f1_score'
                }
                df.rename(columns=column_mapping, inplace=True)
                
                # Create summary of averages per run
                metrics_to_aggregate = ['accuracy', 'Precision', 'Recall', 'Specificity', 'MCC', 'f1_score']
                available_metrics = [col for col in metrics_to_aggregate if col in df.columns]
                agg_dict = {metric: ['mean', 'std'] for metric in available_metrics}
                
                run_summary = df.groupby('Run').agg(agg_dict).reset_index()
                
                # Flatten multi-level column names
                run_summary.columns = ['_'.join(col).strip('_') for col in run_summary.columns.values]
                run_summary.to_csv(model_dir / "run_performance_summary.csv", index=False)
                
                if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
                    df.to_csv(model_dir / "complete_metrics.csv", index=False)
        
        # Generate final combined visualization
        try:
            # Create overall model comparison across all runs
            plt.figure(figsize=(14, 10))
            
            # Prepare data for plotting
            model_names = []
            accuracies = []
            acc_stds = []
            f1_scores = []
            f1_stds = []
            
            for model_type, metrics_list in model_metrics.items():
                if metrics_list:
                    # Calculate mean and std for each metric
                    acc_values = [m.get('accuracy', 0) for m in metrics_list]
                    f1_values = [m.get('f1_score', 0) for m in metrics_list]
                    
                    model_names.append(model_type)
                    accuracies.append(np.mean(acc_values))
                    acc_stds.append(np.std(acc_values))
                    f1_scores.append(np.mean(f1_values))
                    f1_stds.append(np.std(f1_values))
            
            # Set up bar positions
            x = np.arange(len(model_names))
            width = 0.35
            
            # Debug info
            console.print(f"[bold]Visualization data:[/bold]")
            console.print(f"Models: {model_names}")
            console.print(f"Accuracies: {accuracies}")
            console.print(f"F1 Scores: {f1_scores}")
            
            # Only create plot if we have data
            if len(model_names) > 0 and len(accuracies) > 0:
                # Create the plot
                fig, ax = plt.subplots(figsize=(12, 8))
                rects1 = ax.bar(x - width/2, accuracies, width, label='accuracy', yerr=acc_stds)
                rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', yerr=f1_stds)
                
                # Add labels and formatting
                ax.set_ylabel('Score')
                ax.set_title('Model Performance Across All Runs')
                ax.set_xticks(x)
                ax.set_xticklabels(model_names)
                ax.legend()
                plt.ylim(0, 1)
                
                # Add value labels on top of bars
                for rect in rects1:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                for rect in rects2:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            else:
                console.print("[bold red]No model performance data available for visualization![/bold red]")
                # Create empty plot with message
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, "No performance data available", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=16)
                ax.set_title('Model Performance Across All Runs')
                ax.set_xlabel('Models')
                ax.set_ylabel('Score')
            
            plt.tight_layout()
            plt.savefig(Path(config.hyperparameter_tuning.output_dir) / "task_concat" / "overall_model_comparison.png", dpi=300)
            plt.close()
            
            # Generate visualizations for each run
            for run_idx in range(min(n_runs, 1 if debug else float('inf'))):
                generate_performance_visualizations(
                    Path(config.hyperparameter_tuning.output_dir) / "task_concat",
                    model_metrics,
                    run_idx
                )
            
        except Exception as e:
            console.print(f"[yellow]Error generating visualizations: {str(e)}[/yellow]")
    
    end_time = time.time()
    execution_time = end_time - start_time
    console.print(f"[bold]Total execution time: {execution_time:.2f} seconds[/bold]")
    
    return overall_results


if __name__ == "__main__":
    main()
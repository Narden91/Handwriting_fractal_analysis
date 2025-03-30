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
import re

sys.dont_write_bytecode = True
from rich import print
from rich.console import Console
from rich.panel import Panel
from src.processing.processing import process_task
from src.majority_vote import run_majority_vote_analysis


console = Console()


def natural_sort_key(path):
    """
    Sort key function that extracts the task number from the filename
    for proper numerical sorting.
    """
    # Extract the task number from the filename (e.g., "TASK_01" -> 1)
    match = re.search(r'TASK_(\d+)', path.name)
    if match:
        return int(match.group(1))
    # Fallback to standard sorting if pattern doesn't match
    return path.name


def set_global_seeds(seed):
    """Set all common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def check_task_completed(base_dir, model_types, run_idx, task_idx):
    """
    Check if all models for a specific task in a run have completed processing.
    
    Parameters:
    -----------
    base_dir : Path
        Base output directory
    model_types : list
        List of model types to check
    run_idx : int
        Run index (0-based)
    task_idx : int
        Task index (0-based)
        
    Returns:
    --------
    bool
        True if all models have completed this task, False otherwise
    """
    for model_type in model_types:
        model_dir = base_dir / model_type / f"run_{run_idx+1}" / f"task_{task_idx+1}"
        
        # Check for metrics file and predictions file
        metrics_file = model_dir / "metrics.json"
        pred_file = model_dir / "predictions.csv"
        
        if not metrics_file.exists() or not pred_file.exists():
            return False
    
    return True


def load_task_results(base_dir, model_types, run_idx, task_idx):
    """
    Load results for a completed task from existing files.
    
    Parameters:
    -----------
    base_dir : Path
        Base output directory
    model_types : list
        List of model types
    run_idx : int
        Run index (0-based)
    task_idx : int
        Task index (0-based)
        
    Returns:
    --------
    dict
        Dictionary with task results for all models
    """
    task_results = {}
    
    for model_type in model_types:
        metrics_file = base_dir / model_type / f"run_{run_idx+1}" / f"task_{task_idx+1}" / "metrics.json"
        
        try:
            # Load metrics from the file
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Construct a minimal task result object that contains the necessary information
            # for downstream processing (especially majority vote)
            task_results[model_type] = {
                'test_accuracy': metrics.get('accuracy', 0),
                'best_params': {},  # Not needed for majority vote
                'best_cv_score': 0,  # Not needed for majority vote
                'feature_importances': None  # Not needed for majority vote
            }
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load metrics for {model_type} (Run {run_idx+1}, Task {task_idx+1}): {e}[/yellow]")
            # Return an empty dict if any model fails to load, to trigger reprocessing
            return {}
    
    return task_results


def check_run_completed(base_dir, model_types, run_idx, task_files, debug=False):
    """
    Check if an entire run has already been completed.
    
    Parameters:
    -----------
    base_dir : Path
        Base output directory
    model_types : list
        List of model types
    run_idx : int
        Run index (0-based)
    task_files : list
        List of task files
    debug : bool
        Whether debug mode is enabled
        
    Returns:
    --------
    bool
        True if the entire run is completed, False otherwise
    """
    for task_idx, _ in enumerate(task_files):
        if not check_task_completed(base_dir, model_types, run_idx, task_idx):
            return False
        
        # In debug mode, only check the first task
        if debug and task_idx == 0:
            break
    
    return True


@hydra.main(config_path="./config", config_name="ml_config", version_base="1.2")
def main(config: DictConfig):
    """Main function to run the ML pipeline."""
    start_time = time.time()
    console.print(Panel("[bold cyan]🔍 ML Fractal Handwriting Analysis[/bold cyan]",
                      title="Starting Analysis", expand=False))

    # Check if force_reprocess flag is set in config
    force_reprocess = config.settings.get('force_reprocess', False)
    if force_reprocess:
        console.print("[bold yellow]Force reprocess flag is set. Will reprocess all tasks even if results exist.[/bold yellow]")

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

    if verbose > 0:
        console.print(db_info.head())

    # Load all task files (files that start with 'TASK_')
    task_files = [f for f in features_path.glob("TASK_*.csv")]
    task_files.sort(key=natural_sort_key)
    console.print(f"Found {len(task_files)} task files for processing")

    # Initialize output directory
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        main_output_dir = Path(config.hyperparameter_tuning.output_dir)        
        main_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        main_output_dir = None
    
    overall_results = {}

    # Process each run
    for run_idx in range(n_runs):
        run_seed = base_seed + run_idx
        console.print(Panel(f"[bold]Starting Run {run_idx+1}/{n_runs} with Seed {run_seed}[/bold]",
                          expand=False))

        run_results = {}
        
        # Check if the entire run has been completed already
        if (not force_reprocess and 
            main_output_dir is not None and 
            check_run_completed(main_output_dir, config.hyperparameter_tuning.models, run_idx, task_files, debug)):
            
            console.print(f"[bold green]Run {run_idx+1} has already been fully processed. Loading results...[/bold green]")
            
            # Load results for all tasks in this run
            for task_idx, _ in enumerate(task_files):
                task_name = task_idx + 1
                task_results = load_task_results(main_output_dir, config.hyperparameter_tuning.models, run_idx, task_idx)
                run_results[task_name] = task_results
                
                # In debug mode, only process the first task
                if debug and task_idx == 0:
                    break
        else:
            # Process each task in this run
            for task_idx, task in enumerate(task_files):
                # Check if this individual task has already been completed
                if (not force_reprocess and 
                    main_output_dir is not None and 
                    check_task_completed(main_output_dir, config.hyperparameter_tuning.models, run_idx, task_idx)):
                    
                    console.print(f"[bold yellow]Task {task_idx+1} in Run {run_idx+1} has already been processed. Loading results...[/bold yellow]")
                    
                    # Load results for this task
                    task_name = task_idx + 1
                    task_results = load_task_results(main_output_dir, config.hyperparameter_tuning.models, run_idx, task_idx)
                    run_results[task_name] = task_results
                else:
                    # Process the task normally
                    console.print(f"[bold blue]Processing Task {task_idx+1} in Run {run_idx+1}...[/bold blue]")
                    task_results = process_task(
                        task, db_info, config, run_seed, run_idx, task_idx)
                    
                    task_name = task_idx + 1
                    run_results[task_name] = task_results
                
                # In debug mode, only process the first task
                if debug and task_idx == 0:
                    console.print("[bold yellow]Debug mode: stopping after first task[/bold yellow]")
                    break
        
        # Store results for this run
        overall_results[f"run_{run_idx+1}"] = run_results

    # Run majority vote analysis if output_dir is specified
    if main_output_dir is not None:
        console.print(Panel("[bold cyan]🔍 Running Majority Vote Analysis[/bold cyan]",
                        title="Starting Majority Vote Analysis", expand=False))
        
        majority_vote_results = run_majority_vote_analysis(config, task_files, db_info)
        
        mv_time = time.time()
        mv_execution_time = mv_time - start_time
        console.print(f"[bold]Majority vote analysis execution time: {mv_execution_time:.2f} seconds[/bold]")
        
        # Add majority vote results to overall results
        for run_key, run_results in majority_vote_results.items():
            if run_key in overall_results:
                overall_results[run_key]['majority_vote'] = run_results

    if main_output_dir is not None:
        model_metrics = {}

        for model_type in config.hyperparameter_tuning.models:
            model_metrics[model_type] = []

        # Loop through all the tasks and runs to collect metrics
        for run_idx in range(n_runs):
            run_seed = base_seed + run_idx

            for task_idx, task in enumerate(task_files):
                task_name = task_idx + 1

                if debug and task_idx > 0:
                    continue

                # For each model type, read the metrics.json file
                for model_type in config.hyperparameter_tuning.models:
                    model_dir = Path(config.hyperparameter_tuning.output_dir) / model_type
                    metrics_file = model_dir / f"run_{run_idx+1}" / f"task_{task_idx+1}" / "metrics.json"

                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)

                            metrics['Run'] = run_idx + 1
                            metrics['Seed'] = run_seed
                            metrics['Task'] = task_name

                            if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
                                metrics['FeatureSelectionMethod'] = config.feature_selection.method
                                metrics['K'] = config.feature_selection.k
                                metrics['ScoreFunc'] = config.feature_selection.get('score_func', 'f_classif')

                            model_metrics[model_type].append(metrics)
                        except Exception as e:
                            console.print(f"[yellow]Error reading metrics for {model_type} (Run {run_idx+1}, Task {task_idx+1}): {e}[/yellow]")

        # Create performance summary CSV for each model
        for model_type, metrics_list in model_metrics.items():
            if metrics_list:
                model_dir = Path(config.hyperparameter_tuning.output_dir) / model_type
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

        # Create comparison between individual tasks and majority vote
        for model_type in config.hyperparameter_tuning.models:
            # Create a combined CSV with both individual task performance and majority vote
            model_dir = Path(config.hyperparameter_tuning.output_dir) / model_type
            mv_dir = model_dir / "majority_vote"
            
            if mv_dir.exists() and (mv_dir / "summary.csv").exists():
                try:
                    # Read majority vote summary
                    mv_summary = pd.read_csv(mv_dir / "summary.csv")
                    
                    # Read individual task performance if available
                    task_summary_file = model_dir / "run_performance_summary.csv"
                    if task_summary_file.exists():
                        task_summary = pd.read_csv(task_summary_file)
                        
                        # Create a comparison DataFrame
                        comparison_data = []
                        
                        # Add task metrics (averaged across runs)
                        for metric in ['Accuracy', 'F1_score', 'Precision', 'Recall']:
                            if f"{metric}_mean" in task_summary.columns:
                                # Average of means across runs
                                mean_val = task_summary[f"{metric}_mean"].mean()
                                std_val = task_summary[f"{metric}_std"].mean()
                                
                                comparison_data.append({
                                    'Metric': metric,
                                    'Source': 'Individual Tasks',
                                    'Mean': mean_val,
                                    'Std': std_val
                                })
                        
                        # Add majority vote metrics
                        for _, row in mv_summary.iterrows():
                            comparison_data.append({
                                'Metric': row['Metric'],
                                'Source': 'Majority Vote',
                                'Mean': row['Mean'],
                                'Std': row['Std']
                            })
                        
                        # Create and save comparison DataFrame
                        comparison_df = pd.DataFrame(comparison_data)
                        comparison_df.to_csv(model_dir / "task_vs_majority_comparison.csv", index=False)
                        
                        # Create visualization of the comparison
                        try:
                            # Plot comparison
                            plt.figure(figsize=(12, 8))
                            
                            # Reshape for plotting
                            plot_data = []
                            for _, row in comparison_df.iterrows():
                                if row['Metric'] in ['Accuracy', 'F1_score']:  # Focus on main metrics
                                    plot_data.append({
                                        'Metric': row['Metric'],
                                        'Source': row['Source'],
                                        'Value': row['Mean'],
                                        'Error': row['Std']
                                    })
                            
                            plot_df = pd.DataFrame(plot_data)
                            
                            # Create grouped bar chart
                            sns.barplot(x='Metric', y='Value', hue='Source', data=plot_df)
                            
                            # Add error bars
                            for i, bar in enumerate(plt.gca().patches):
                                row = plot_df.iloc[i]
                                plt.errorbar(
                                    bar.get_x() + bar.get_width()/2,
                                    row['Value'],
                                    yerr=row['Error'],
                                    fmt='none', 
                                    color='black', 
                                    capsize=5
                                )
                            
                            plt.title(f'Task vs Majority Vote Performance: {model_type.upper()}')
                            plt.ylim(0, 1)
                            plt.grid(axis='y', linestyle='--', alpha=0.7)
                            
                            # Add value labels on top of bars
                            for i, bar in enumerate(plt.gca().patches):
                                plt.text(
                                    bar.get_x() + bar.get_width()/2,
                                    bar.get_height() + 0.01,
                                    f'{plot_df.iloc[i]["Value"]:.3f}',
                                    ha='center',
                                    fontsize=9
                                )
                            
                            plt.tight_layout()
                            plt.savefig(model_dir / "task_vs_majority_comparison.png", dpi=300)
                            plt.close()
                        except Exception as e:
                            console.print(f"[yellow]Error creating comparison visualization: {str(e)}[/yellow]")
                        
                        console.print(f"[bold green]Created task vs majority vote comparison for {model_type}[/bold green]")
                except Exception as e:
                    console.print(f"[yellow]Error creating comparison for {model_type}: {str(e)}[/yellow]")

    end_time = time.time()
    execution_time = end_time - start_time
    console.print(f"[bold]Total execution time: {execution_time:.2f} seconds[/bold]")

    return overall_results


if __name__=="__main__":
    main()
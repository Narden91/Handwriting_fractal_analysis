import hydra
from omegaconf import DictConfig
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

sys.dont_write_bytecode = True
from rich import print
from rich.console import Console
from rich.panel import Panel
from src.hp_tuning import run_hyperparameter_search
from src.processing.processing import process_task


console = Console()


@hydra.main(config_path="./config", config_name="ml_config", version_base="1.2")
def main(config: DictConfig):
    """Main function to run the ML pipeline."""
    start_time = time.time()
    
    console.print(Panel("[bold cyan]🔍 ML Fractal Handwriting Analysis[/bold cyan]", 
                      title="Starting Analysis", expand=False))
    
    # Print feature selection information if enabled
    if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
        fs_method = config.feature_selection.method
        k_value = config.feature_selection.k
        score_func = config.feature_selection.get('score_func', 'f_classif')
        console.print(Panel(f"[bold magenta]Feature Selection: {fs_method.upper()} (k={k_value}, scoring={score_func})[/bold magenta]", 
                          expand=False))
    
    verbose = config.settings.verbose
    debug = config.settings.debug
    base_seed = config.settings.base_seed
    n_runs = config.settings.n_runs
    
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
    console.print(f"Found {len(task_files)} task files for processing")
    
    # Create main results directory if specified
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        main_output_dir = Path(config.hyperparameter_tuning.output_dir)
        main_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create feature selection directory if feature selection is enabled
        if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
            fs_dir = main_output_dir / "feature_selection"
            fs_dir.mkdir(parents=True, exist_ok=True)
    
    # Store overall results across runs and tasks
    overall_results = {}
    
    # Perform multiple runs with different seeds
    for run_idx in range(n_runs):
        # Generate a unique seed for this run
        run_seed = base_seed + run_idx * 100  # Ensuring distinct seeds
        
        console.print(Panel(f"[bold]Starting Run {run_idx+1}/{n_runs} with Seed {run_seed}[/bold]", 
                          expand=False))
        
        run_results = {}
        
        # Process each task
        for task_idx, task in enumerate(task_files):
            task_results = process_task(
                task, db_info, config, run_seed, run_idx, task_idx)
            
            # Store results for all models
            task_name = task_idx + 1
            run_results[task_name] = task_results
            
            # If in debug mode, only process the first task
            if debug and task_idx == 0:
                console.print("[bold yellow]Debug mode: stopping after first task[/bold yellow]")
                break
        
        # Store run results
        overall_results[f"run_{run_idx+1}"] = run_results
            
    if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
        model_metrics = {}
        
        for model_type in config.hyperparameter_tuning.models:
            model_metrics[model_type] = []
        
        # Loop through all the tasks and runs to collect metrics
        for run_idx in range(n_runs):
            run_key = f"run_{run_idx+1}"
            run_seed = base_seed + run_idx * 100
            
            for task_idx, task in enumerate(task_files):
                task_name = task_idx + 1
                
                if debug and task_idx > 0:
                    continue
                    
                # For each model type, read the metrics.json file
                for model_type in config.hyperparameter_tuning.models:
                    model_dir = Path(config.hyperparameter_tuning.output_dir) / model_type
                    metrics_file = model_dir / f"run_{run_idx+1}" / f"task_{task_idx+1}" / "metrics.json"
                    
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        metrics['Run'] = run_idx + 1
                        metrics['Seed'] = run_seed
                        metrics['Task'] = task_name
                        
                        # Add feature selection info if enabled
                        if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
                            metrics['FeatureSelectionMethod'] = config.feature_selection.method
                            metrics['K'] = config.feature_selection.k
                            metrics['ScoreFunc'] = config.feature_selection.get('score_func', 'f_classif')
                        
                        model_metrics[model_type].append(metrics)
        
        # Create only run_performance_summary.csv for each model
        for model_type, metrics_list in model_metrics.items():
            if metrics_list:
                model_dir = Path(config.hyperparameter_tuning.output_dir) / model_type
                df = pd.DataFrame(metrics_list)
                
                # Rename columns
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
                
                # Group by Run and calculate statistics
                run_summary = df.groupby('Run').agg(agg_dict).reset_index()
                
                # Flatten multi-level column names
                run_summary.columns = ['_'.join(col).strip('_') for col in run_summary.columns.values]
                
                # Save run summary
                run_summary.to_csv(model_dir / "run_performance_summary.csv", index=False)

                # If feature selection was used, create a summary comparing performance with feature selection
                if hasattr(config, 'feature_selection') and config.feature_selection.enabled:
                    # Save the complete metrics for analysis
                    df.to_csv(model_dir / "complete_metrics.csv", index=False)
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    console.print(f"[bold]Total execution time: {execution_time:.2f} seconds[/bold]")
    
    return overall_results


if __name__=="__main__":
    main()
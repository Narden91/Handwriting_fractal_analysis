import hydra
from omegaconf import DictConfig
import sys
import pandas as pd
from pathlib import Path
import pickle
import logging

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
sys.dont_write_bytecode = True
from rich import print
from src.hp_tuning import run_hyperparameter_search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(config_path="./config", config_name="ml_config", version_base="1.2")
def main(config: DictConfig):
    
    # --- SETUP ---
    # Display header
    print(f"🔍 [bold]ML Fractal Handwriting Analysis[/bold]")
    verbose = config.settings.verbose
    
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
    print(db_info.head()) if verbose > 0 else None
    
    # Load csv file that starts with 'TASK_'
    task_files = [f for f in features_path.glob("TASK_*.csv")]
    
    for i, task in enumerate(task_files):
        print(f"[bold]Processing Task {i+1}: {task.name}[/bold]")
        task_df = pd.read_csv(task)
        
        # Left join with db_info
        task_df = task_df.merge(db_info, on="Id", how="left")
        
        # Move Class column to the end
        class_col = task_df.pop("Class")
        task_df["Class"] = class_col
        
        print(task_df.head()) if verbose > 0 else None
        
        # --- ML ANALYSIS ---
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            task_df.drop("Class", axis=1), task_df["Class"], test_size=0.2, random_state=42, stratify=task_df["Class"])
        
        X_train = X_train.drop("Id", axis=1)
        X_test = X_test.drop("Id", axis=1)
        
        if verbose > 0:
            print(f"X_train: \n {X_train.shape}")
            print(f"y_train: \n {y_train.shape}")
            print(f"X_test: \n {X_test.shape}")
            print(f"y_test: \n {y_test.shape}")
        
        # Scale data
        scaler = StandardScaler() if config.preprocessing.scaler == "Standard" else RobustScaler()
        
        # Scale data but not the Work and Sex columns
        # Extract binary features
        X_train_no_scale = X_train[["Work", "Sex"]]
        X_test_no_scale = X_test[["Work", "Sex"]]

        # Remove binary features from the datasets to be scaled
        X_train_to_scale = X_train.drop(["Work", "Sex"], axis=1)
        X_test_to_scale = X_test.drop(["Work", "Sex"], axis=1)

        # Store column names for later use
        scaled_columns = X_train_to_scale.columns

        # Scale the non-binary features
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
        
        print(X_train.head()) if verbose > 0 else None
        
        # --- MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION ---
        print("[bold]Starting hyperparameter optimization...[/bold]")
        
        # Create task-specific output directory
        task_output_dir = None
        if config.hyperparameter_tuning.output_dir:
            task_output_dir = Path(config.hyperparameter_tuning.output_dir) / f"task_{i+1}"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the preprocessed datasets for reference
            X_train.to_csv(task_output_dir / "X_train.csv", index=False)
            X_test.to_csv(task_output_dir / "X_test.csv", index=False)
            y_train.to_csv(task_output_dir / "y_train.csv", index=False)
            y_test.to_csv(task_output_dir / "y_test.csv", index=False)
        
        # Update config with task-specific output directory
        task_config = config.copy()
        if hasattr(config.hyperparameter_tuning, 'output_dir') and config.hyperparameter_tuning.output_dir:
            task_config.hyperparameter_tuning.output_dir = str(task_output_dir)
        
        # Run hyperparameter optimization
        results = run_hyperparameter_search(task_config, X_train, y_train, X_test, y_test)
        
        # Save results
        if task_output_dir:
            # Save best models
            for model_type, result in results.items():
                with open(task_output_dir / f"{model_type}_best_model.pkl", "wb") as f:
                    pickle.dump(result['model'], f)
                
                # Save best parameters
                with open(task_output_dir / f"{model_type}_best_params.txt", "w") as f:
                    f.write(f"Best CV score: {result['best_cv_score']:.4f}\n")
                    f.write(f"Test score: {result['test_score']:.4f}\n\n")
                    f.write("Best parameters:\n")
                    for param, value in result['best_params'].items():
                        f.write(f"{param}: {value}\n")
        
        # Find best model overall
        best_model_type = max(results, key=lambda k: results[k]['test_score'])
        best_model_score = results[best_model_type]['test_score']
        
        print(f"[bold green]Best model for Task {i+1}: {best_model_type} with accuracy {best_model_score:.4f}[/bold green]")
        print("-" * 80)
        
        if i == 0 and 'debug' in config and config.debug:
            print("[bold yellow]Debug mode: stopping after first task[/bold yellow]")
            break
        
    return


if __name__=="__main__":
    main()
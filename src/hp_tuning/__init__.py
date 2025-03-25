"""
Module for hyperparameter optimization using Optuna.
Provides optimization for Random Forest, LightGBM, and MLP classifiers.
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgbm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging
from rich import print
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)

def objective_rf(trial, X_train, y_train, X_val, y_val, metric='accuracy'):
    """
    Objective function for Random Forest optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize ('accuracy', 'f1', 'roc_auc')
        
    Returns:
        score: Metric score to be maximized
    """
    # Define hyperparameters to search
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
    
    # Create and train the model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        class_weight=class_weight,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate based on the chosen metric
    if metric == 'accuracy':
        score = accuracy_score(y_val, clf.predict(X_val))
    elif metric == 'f1':
        score = f1_score(y_val, clf.predict(X_val), average='weighted')
    elif metric == 'roc_auc':
        try:
            y_pred_proba = clf.predict_proba(X_val)
            score = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        except:
            # Fallback to accuracy if ROC AUC fails
            score = accuracy_score(y_val, clf.predict(X_val))
    
    return score

def objective_lgbm(trial, X_train, y_train, X_val, y_val, metric='accuracy'):
    """
    Objective function for LightGBM optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize ('accuracy', 'f1', 'roc_auc')
        
    Returns:
        score: Metric score to be maximized
    """
    # Define hyperparameters to search
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    num_leaves = trial.suggest_int('num_leaves', 20, 150)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 100)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    
    # Create and train the model
    clf = lgbm.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate based on the chosen metric
    if metric == 'accuracy':
        score = accuracy_score(y_val, clf.predict(X_val))
    elif metric == 'f1':
        score = f1_score(y_val, clf.predict(X_val), average='weighted')
    elif metric == 'roc_auc':
        try:
            y_pred_proba = clf.predict_proba(X_val)
            score = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        except:
            # Fallback to accuracy if ROC AUC fails
            score = accuracy_score(y_val, clf.predict(X_val))
    
    return score

def objective_mlp(trial, X_train, y_train, X_val, y_val, metric='accuracy'):
    """
    Objective function for Multi-layer Perceptron optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize ('accuracy', 'f1', 'roc_auc')
        
    Returns:
        score: Metric score to be maximized
    """
    # Define hyperparameters to search
    hidden_layer_sizes = []
    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        hidden_layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 32, 256))
    
    learning_rate_init = trial.suggest_float('learning_rate_init', 0.0001, 0.1, log=True)
    alpha = trial.suggest_float('alpha', 0.0001, 0.01, log=True)
    batch_size = trial.suggest_categorical('batch_size', ['auto', 64, 128, 256])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    
    # Create and train the model
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        batch_size=batch_size,
        activation=activation,
        solver=solver,
        max_iter=500,  # Higher max_iter to ensure convergence
        random_state=42
    )
    
    try:
        clf.fit(X_train, y_train)
        
        # Evaluate based on the chosen metric
        if metric == 'accuracy':
            score = accuracy_score(y_val, clf.predict(X_val))
        elif metric == 'f1':
            score = f1_score(y_val, clf.predict(X_val), average='weighted')
        elif metric == 'roc_auc':
            try:
                y_pred_proba = clf.predict_proba(X_val)
                score = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
            except:
                # Fallback to accuracy if ROC AUC fails
                score = accuracy_score(y_val, clf.predict(X_val))
    except:
        # Return a very low score if the model fails to converge
        return -1.0
    
    return score

def optimize_hyperparameters(model_type, X_train, y_train, n_trials=100, cv=5, metric='accuracy', verbose=0):
    """
    Optimize hyperparameters for the given model type.
    
    Args:
        model_type: 'rf', 'lgbm', or 'mlp'
        X_train: Training features
        y_train: Training labels
        n_trials: Number of trials for Optuna
        cv: Number of cross-validation folds
        metric: Metric to optimize ('accuracy', 'f1', 'roc_auc')
        verbose: Verbosity level
        
    Returns:
        best_params: Dictionary of best parameters
        best_score: Best score achieved
        best_model: Trained model with best parameters
    """
    # Create cross-validation splitter
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Define the objective function for cross-validation
    def objective_cv(trial):
        # Choose model and get objective function
        if model_type == 'rf':
            model_cls = RandomForestClassifier
            param_keys = ['n_estimators', 'max_depth', 'min_samples_split', 
                          'min_samples_leaf', 'bootstrap', 'class_weight']
            
            # Define hyperparameters to search
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': 42
            }
            
        elif model_type == 'lgbm':
            model_cls = lgbm.LGBMClassifier
            param_keys = ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth',
                          'min_child_samples', 'subsample', 'colsample_bytree']
            
            # Define hyperparameters to search
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42
            }
            
        elif model_type == 'mlp':
            model_cls = MLPClassifier
            
            # Define hyperparameters to search
            hidden_layer_sizes = []
            n_layers = trial.suggest_int('n_layers', 1, 3)
            for i in range(n_layers):
                hidden_layer_sizes.append(trial.suggest_int(f'n_units_l{i}', 32, 256))
            
            params = {
                'hidden_layer_sizes': tuple(hidden_layer_sizes),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1, log=True),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                'batch_size': trial.suggest_categorical('batch_size', ['auto', 64, 128, 256]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
                'max_iter': 500,  # Higher max_iter to ensure convergence
                'random_state': 42
            }
            param_keys = list(params.keys())
            param_keys.remove('random_state')
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create the model with trial parameters
        model = model_cls(**params)
        
        # Cross-validation
        if metric == 'accuracy':
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        elif metric == 'f1':
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
        elif metric == 'roc_auc':
            try:
                scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc_ovr')
            except:
                # Fallback to accuracy if ROC AUC fails
                scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        
        # Return the mean of the cross-validation scores
        return scores.mean()
    
    # Create the Optuna study
    study = optuna.create_study(direction='maximize')
    
    # Set optuna verbosity
    optuna_verbosity = 1 if verbose > 0 else 0
    
    # Run the optimization
    study.optimize(objective_cv, n_trials=n_trials, show_progress_bar=True, verbose=optuna_verbosity)
    
    # Get best parameters
    best_params = study.best_params
    
    if verbose > 0:
        print(f"[bold]Best parameters for {model_type}:[/bold]")
        for param, value in best_params.items():
            print(f"{param}: {value}")
    
    # Create the best model
    if model_type == 'rf':
        best_model = RandomForestClassifier(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', None),
            min_samples_split=best_params.get('min_samples_split', 2),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            bootstrap=best_params.get('bootstrap', True),
            class_weight=best_params.get('class_weight', None),
            random_state=42
        )
    elif model_type == 'lgbm':
        best_model = lgbm.LGBMClassifier(
            n_estimators=best_params.get('n_estimators', 100),
            learning_rate=best_params.get('learning_rate', 0.1),
            num_leaves=best_params.get('num_leaves', 31),
            max_depth=best_params.get('max_depth', -1),
            min_child_samples=best_params.get('min_child_samples', 20),
            subsample=best_params.get('subsample', 1.0),
            colsample_bytree=best_params.get('colsample_bytree', 1.0),
            random_state=42
        )
    elif model_type == 'mlp':
        hidden_layer_sizes = []
        n_layers = best_params.get('n_layers', 1)
        for i in range(n_layers):
            hidden_layer_sizes.append(best_params.get(f'n_units_l{i}', 100))
            
        best_model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            learning_rate_init=best_params.get('learning_rate_init', 0.001),
            alpha=best_params.get('alpha', 0.0001),
            batch_size=best_params.get('batch_size', 'auto'),
            activation=best_params.get('activation', 'relu'),
            solver=best_params.get('solver', 'adam'),
            max_iter=500,
            random_state=42
        )
    
    # Train the model on the full training set
    best_model.fit(X_train, y_train)
    
    # Optionally plot optimization history
    if verbose > 1:
        # Plot optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        plt.title(f"Optimization History for {model_type}")
        plt.show()
        
        # Plot parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        plt.title(f"Parameter Importances for {model_type}")
        plt.show()
    
    return best_params, study.best_value, best_model

def evaluate_model(model, X_test, y_test, model_name, output_dir=None, verbose=0):
    """
    Evaluate a model on test data and optionally save results.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for reporting
        output_dir: Directory to save results (if None, don't save)
        verbose: Verbosity level
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Try to get ROC AUC score for multi-class
    try:
        y_pred_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except:
        roc_auc = None
    
    # Print results if verbose
    if verbose > 0:
        print(f"[bold]{model_name} Evaluation:[/bold]")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC Score (OVR): {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sorted(set(y_test)),
                    yticklabels=sorted(set(y_test)))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if output_dir:
            plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
        
        plt.show()
    
    # Save results if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classification report
        pd.DataFrame(report).transpose().to_csv(
            output_dir / f"{model_name}_classification_report.csv")
        
        # Save feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create DataFrame for feature importances
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': [importances[i] for i in indices]
            })
            
            # Save to CSV
            importance_df.to_csv(output_dir / f"{model_name}_feature_importances.csv", index=False)
            
            # Plot feature importances
            if verbose > 0:
                plt.figure(figsize=(12, 8))
                plt.title(f'Feature Importances - {model_name}')
                plt.bar(range(len(indices[:20])), 
                        [importances[i] for i in indices[:20]],
                        align='center')
                plt.xticks(range(len(indices[:20])), 
                          [feature_names[i] for i in indices[:20]], 
                          rotation=90)
                plt.tight_layout()
                plt.savefig(output_dir / f"{model_name}_feature_importances.png")
                plt.show()
    
    # Return results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    }
    
    return results

def run_hyperparameter_search(config, X_train, y_train, X_test, y_test):
    """
    Run hyperparameter search for the specified models.
    
    Args:
        config: Configuration dictionary (from Hydra)
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        results: Dictionary with results for each model
    """
    results = {}
    verbose = config.settings.verbose
    
    # Get models to optimize from config
    models_to_optimize = config.hyperparameter_tuning.models
    n_trials = config.hyperparameter_tuning.n_trials
    metric = config.hyperparameter_tuning.metric
    cv = config.hyperparameter_tuning.cv
    
    # Output directory for saving results
    output_dir = config.hyperparameter_tuning.output_dir
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_type in models_to_optimize:
        print(f"[bold]Optimizing hyperparameters for {model_type}...[/bold]")
        
        best_params, best_score, best_model = optimize_hyperparameters(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            n_trials=n_trials,
            cv=cv,
            metric=metric,
            verbose=verbose
        )
        
        # Evaluate on test set
        print(f"[bold]Evaluating {model_type} on test set...[/bold]")
        eval_results = evaluate_model(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_type,
            output_dir=output_dir,
            verbose=verbose
        )
        
        # Store results
        results[model_type] = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_score': eval_results['accuracy'],
            'model': best_model,
            'evaluation': eval_results
        }
        
        print(f"[bold]Best CV score for {model_type}:[/bold] {best_score:.4f}")
        print(f"[bold]Test score for {model_type}:[/bold] {eval_results['accuracy']:.4f}")
        print("-" * 50)
    
    return results
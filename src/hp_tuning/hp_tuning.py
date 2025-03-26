import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgbm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import matthews_corrcoef 
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from contextlib import contextmanager
import json

# Setup loggers
logger = logging.getLogger(__name__)
optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.ERROR)
lgbm_logger = logging.getLogger('lightgbm')
lgbm_logger.setLevel(logging.ERROR)
xgb_logger = logging.getLogger('xgboost')
xgb_logger.setLevel(logging.ERROR)


# Suppress stdout/stderr temporarily
@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def optimize_hyperparameters(model_type, X_train, y_train, n_trials=100, cv=5, metric='accuracy', run_seed=42):
    """Optimize hyperparameters using cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=run_seed)

    def objective_cv(trial):
        if model_type == 'rf':
            model_cls = RandomForestClassifier
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'random_state': run_seed
            }

        elif model_type == 'xgbrf': 
            model_cls = xgb.XGBRFClassifier
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # 
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1, log=True),
                'random_state': run_seed,
                'n_jobs': -1,
                'verbosity': 0,
                'objective': 'binary:logistic'
            }

        elif model_type == 'dt':
            model_cls = DecisionTreeClassifier
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'random_state': run_seed
            }

        elif model_type == 'svc':
            model_cls = SVC
            params = {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']) != 'linear' else 'scale', 
                'degree': trial.suggest_int('degree', 2, 5) if trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']) == 'poly' else 3, 
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'random_state': run_seed,
                'probability': True 
            }

        elif model_type == 'lgbm':
            model_cls = lgbm.LGBMClassifier
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': run_seed,
                'deterministic': True,
                'force_col_wise': True,
                'verbose': -1,
                'silent': True
            }

        elif model_type == 'mlp':
            model_cls = MLPClassifier
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
                'max_iter': 500,
                'random_state': run_seed
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model = model_cls(**params)

        if metric == 'accuracy':
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy', error_score='raise') 
        elif metric == 'f1':
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1', error_score='raise') 

        return scores.mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=run_seed),
    )

    with suppress_stdout_stderr():
        study.optimize(
            objective_cv,
            n_trials=n_trials,
            show_progress_bar=False,
            n_jobs=1  
        )

    return study.best_params, study.best_value

def evaluate_model(model_type, best_params, X_train, y_train, X_test, y_test, output_dir=None, run_seed=42):
    """
    Train a model with best parameters and evaluate on test data.

    Args:
        model_type: 'rf', 'lgbm', 'mlp', 'xgb', 'dt', or 'svc'
        best_params: Best parameters found by Optuna
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Dictionary with directories for each model or None
        run_seed: Random seed for reproducibility

    Returns:
        results: Dictionary with evaluation metrics and feature importances
    """
    # Get model directory from the dictionary if provided
    if output_dir and isinstance(output_dir, dict) and model_type in output_dir:
        model_dir = output_dir[model_type]
    elif output_dir and not isinstance(output_dir, dict):
        model_dir = Path(output_dir)
    else:
        model_dir = None

    # Create and train the model with best parameters
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', None),
            min_samples_split=best_params.get('min_samples_split', 2),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            bootstrap=best_params.get('bootstrap', True),
            class_weight=best_params.get('class_weight', None),
            random_state=run_seed  
        )
    elif model_type == 'xgbrf': # XGBoost Random Forest
        model = xgb.XGBRFClassifier(
            n_estimators=best_params.get('n_estimators', 100),
            max_depth=best_params.get('max_depth', None),
            min_samples_split=best_params.get('min_samples_split', 2),
            min_child_weight=best_params.get('min_child_weight', 1), 
            colsample_bynode=best_params.get('colsample_bynode', 0.8),
            reg_alpha=best_params.get('reg_alpha', 0),
            reg_lambda=best_params.get('reg_lambda', 1),
            random_state=run_seed,  
            n_jobs=-1,
            verbosity=0,
            objective='binary:logistic'
        )
    elif model_type == 'dt':
        model = DecisionTreeClassifier(
            max_depth=best_params.get('max_depth', None),
            min_samples_split=best_params.get('min_samples_split', 2),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            class_weight=best_params.get('class_weight', None),
            criterion=best_params.get('criterion', 'gini'),
            random_state=run_seed  
        )
    elif model_type == 'svc':
        model = SVC(
            C=best_params.get('C', 1.0),
            kernel=best_params.get('kernel', 'rbf'),
            gamma=best_params.get('gamma', 'scale'),
            degree=best_params.get('degree', 3),
            class_weight=best_params.get('class_weight', None),
            random_state=run_seed,  
            probability=True
        )
    elif model_type == 'lgbm':
        model = lgbm.LGBMClassifier(
            n_estimators=best_params.get('n_estimators', 100),
            learning_rate=best_params.get('learning_rate', 0.1),
            num_leaves=best_params.get('num_leaves', 31),
            max_depth=best_params.get('max_depth', -1),
            min_child_samples=best_params.get('min_child_samples', 20),
            subsample=best_params.get('subsample', 1.0),
            colsample_bytree=best_params.get('colsample_bytree', 1.0),
            random_state=run_seed,  
            verbose=-1,
            silent=True
        )
    elif model_type == 'mlp':
        hidden_layer_sizes = []
        n_layers = best_params.get('n_layers', 1)
        for i in range(n_layers):
            hidden_layer_sizes.append(best_params.get(f'n_units_l{i}', 100))

        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            learning_rate_init=best_params.get('learning_rate_init', 0.001),
            alpha=best_params.get('alpha', 0.0001),
            batch_size=best_params.get('batch_size', 'auto'),
            activation=best_params.get('activation', 'relu'),
            solver=best_params.get('solver', 'adam'),
            max_iter=500,
            random_state=run_seed  
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    with suppress_stdout_stderr():
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics for binary classification
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')

    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    if len(cm) == 2:
        tn, fp = cm[0][0], cm[0][1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0

    # Try to get feature importances if available
    feature_importances = None
    if hasattr(model, 'feature_importances_'):
        feature_names = X_train.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        feature_importances = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': [importances[i] for i in indices]
        })

        # Save feature importances if model_dir is provided
        if model_dir:
            feature_importances.to_csv(model_dir / "feature_importances.csv", index=False)
    elif model_type == 'svc':
        if best_params.get('kernel') == 'linear' and hasattr(model, 'coef_'): 
            feature_names = X_train.columns
            importances = np.abs(model.coef_[0]) 
            indices = np.argsort(importances)[::-1]

            feature_importances = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': [importances[i] for i in indices]
            })

            if model_dir:
                feature_importances.to_csv(model_dir / "feature_importances.csv", index=False)

    if model_dir:
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'mcc': float(mcc),
            'f1_score': float(f1)
        }
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    results = {
        'model_type': model_type,
        'best_params': best_params,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'mcc': mcc,
        'f1_score': f1,
        'feature_importances': feature_importances
    }

    return results

def run_hyperparameter_search(models_to_optimize, X_train, y_train, X_test, y_test,
                             n_trials=100, cv=5, metric='accuracy', output_dir=None, run_seed=42):
    """Run hyperparameter search for the specified models."""
    results = {}

    for model_type in models_to_optimize:
        best_params, best_cv_score = optimize_hyperparameters(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            n_trials=n_trials,
            cv=cv,
            metric=metric,
            run_seed=run_seed
        )

        eval_results = evaluate_model(
            model_type=model_type,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
            run_seed=run_seed
        )

        results[model_type] = {
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'test_accuracy': eval_results['accuracy'],
            'feature_importances': eval_results['feature_importances']
        }

    return results
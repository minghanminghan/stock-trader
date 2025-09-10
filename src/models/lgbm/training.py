import lightgbm as lgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from src.utils.logging_config import logger

def train_model_walk_forward(X, y):
    """
    Trains a LightGBM classifier model using walk-forward validation.

    Args:
        X (pd.DataFrame): DataFrame of features.
        y (pd.Series): Series of labels.

    Returns:
        (lgb.LGBMClassifier, dict): The trained model and a dictionary of average metrics.
    """
    logger.info("Starting walk-forward model training...")

    tscv = TimeSeriesSplit(n_splits=5)
    all_preds = []
    all_true = []
    model = None  # Initialize model to None
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        logger.info(f"--- Fold {fold+1}/5 ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")

        model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test).tolist()
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)

    logger.info("--- Overall Walk-Forward Performance ---")
    accuracy = accuracy_score(all_true, all_preds)
    report = classification_report(all_true, all_preds)

    logger.info(f"Overall Model Accuracy: {accuracy:.4f}")
    logger.info(f"Overall Classification Report: \n{report}")
    
    logger.info("Model training complete.")
    
    # Return the last trained model and the overall metrics
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }
    
    return model, metrics

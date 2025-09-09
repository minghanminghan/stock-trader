import pandas as pd
import os
import pickle
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

from src.config import TICKERS, START_DATE, END_DATE, DATA_DIR
from src.data_ingestion import fetch_stock_data
from src.feature_engineering import compute_features
from src.labeling import create_labels
from src.models.training import train_model_walk_forward
from src.models.evaluation import evaluate_strategy
from src.utils.logging_config import logger

def run_pipeline():
    """
    Runs the full ML pipeline from data ingestion to model evaluation.
    """
    logger.info("Starting the ML training pipeline...")

    # 1. Data Ingestion
    fetch_stock_data(tickers=TICKERS, start_date=START_DATE, end_date=END_DATE)

    # 2. Load Data
    all_data = []
    for ticker in TICKERS:
        file_path = os.path.join(DATA_DIR, f"{ticker}_1min_{START_DATE}_to_{END_DATE}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            all_data.append(df)
    
    if not all_data:
        logger.error("No data found. Exiting pipeline.")
        return

    combined_df = pd.concat(all_data)

    # 3. Feature Engineering
    featured_df = compute_features(combined_df)

    # 4. Labeling
    labeled_df = create_labels(featured_df)

    # 5. Model Training
    features = [
        'return_1m', 'mom_5m', 'mom_15m', 'mom_60m', 
        'vol_15m', 'vol_60m', 'vol_zscore', 
        'time_sin', 'time_cos'
    ]
    target = 'label'

    X = labeled_df[features]
    y = labeled_df[target]

    # Note: train_model_walk_forward now returns the last trained model and metrics
    model, metrics = train_model_walk_forward(X, y)

    # 6. Evaluation (comprehensive walk-forward validation results)
    if model:
        logger.info("=== Walk-Forward Validation Results ===")
        logger.info(f"Overall Model Accuracy: {metrics['accuracy']:.4f}")
        logger.info("Classification Report:")
        logger.info(f"\n{metrics['classification_report']}")
        
        # Optional: Strategy evaluation on last fold for trading metrics
        logger.info("=== Strategy Evaluation (Last Fold) ===")
        tscv = TimeSeriesSplit(n_splits=5)
        last_train_index, last_test_index = list(tscv.split(X))[-1]
        
        X_test = X.iloc[last_test_index]
        y_pred = model.predict(X_test)
        fwd_returns = labeled_df['fwd_return'].loc[X_test.index]

        evaluate_strategy(X_test, y_pred, fwd_returns)
        
        # 7. Save Model Weights
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join("src", "models", "weights", f"lgbm_model_{timestamp}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

    else:
        logger.error("Model training failed, skipping evaluation.")

    logger.info("ML training pipeline finished successfully.")

if __name__ == "__main__":
    run_pipeline()

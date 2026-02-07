# Production Pipelines

## Overview

Production ML pipelines for quantitative finance orchestrate the end-to-end workflow from data ingestion through model training, validation, and deployment. These pipelines must handle the unique challenges of financial data: market calendars, corporate actions, regulatory requirements, and real-time data feeds.

## Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Ingest │────▶│   Feature    │────▶│    Model     │
│  (Market Data)│     │  Engineering │     │   Training   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Production  │◀────│   Model      │◀────│  Validation  │
│  Deployment  │     │  Registry    │     │  & Backtest  │
└──────────────┘     └──────────────┘     └──────────────┘
        │
        ▼
┌──────────────┐
│  Monitoring  │
│  & Alerting  │
└──────────────┘
```

## Implementation

```python
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
import logging

@dataclass
class PipelineConfig:
    """Configuration for production ML pipeline."""
    model_name: str
    universe: str  # 'sp500', 'russell2000', etc.
    training_window_days: int = 756  # ~3 years
    validation_window_days: int = 252  # ~1 year
    retraining_frequency: str = 'monthly'
    min_sharpe_threshold: float = 0.5
    max_drawdown_threshold: float = -0.15
    model_registry_uri: str = 'sqlite:///models.db'


class ProductionPipeline:
    """End-to-end ML pipeline for quantitative finance."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(config.model_name)
    
    def run(self, as_of_date: date):
        """Execute full pipeline for a given date."""
        self.logger.info(f"Starting pipeline for {as_of_date}")
        
        # Step 1: Data preparation
        train_data, val_data, test_data = self._prepare_data(as_of_date)
        
        # Step 2: Feature engineering
        features_train = self._compute_features(train_data)
        features_val = self._compute_features(val_data)
        
        # Step 3: Model training
        model = self._train_model(features_train)
        
        # Step 4: Validation
        metrics = self._validate_model(model, features_val)
        
        # Step 5: Quality gate
        if self._passes_quality_gate(metrics):
            # Step 6: Register and deploy
            self._register_model(model, metrics, as_of_date)
            self._deploy_model(model)
            self.logger.info(f"Model deployed: Sharpe={metrics['sharpe']:.2f}")
        else:
            self.logger.warning(f"Model failed quality gate: {metrics}")
    
    def _passes_quality_gate(self, metrics: dict) -> bool:
        return (metrics.get('sharpe', 0) >= self.config.min_sharpe_threshold and
                metrics.get('max_drawdown', -1) >= self.config.max_drawdown_threshold)
```

## Scheduling

```python
# Example: Monthly retraining with Airflow-style scheduling
from datetime import timedelta

class PipelineScheduler:
    def __init__(self, pipeline, schedule='monthly'):
        self.pipeline = pipeline
        self.schedule = schedule
    
    def get_next_run_dates(self, start: date, end: date):
        """Generate retraining dates respecting market calendar."""
        dates = []
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:
                dates.append(current)
            if self.schedule == 'monthly':
                # First business day of each month
                if current.month != (current + timedelta(days=1)).month:
                    next_month = current + timedelta(days=1)
                    while next_month.weekday() >= 5:
                        next_month += timedelta(days=1)
                    dates.append(next_month)
            current += timedelta(days=1)
        return dates
```

## Best Practices

- **Automate the full pipeline** — manual steps introduce errors and delays
- **Use point-in-time data** everywhere to prevent look-ahead bias
- **Implement quality gates** with clear, pre-defined thresholds
- **Version everything**: data snapshots, features, models, and configs
- **Monitor model drift** and trigger retraining when performance degrades
- **Maintain audit trails** for regulatory compliance
- **Test the pipeline itself** — not just the model

## References

1. De Prado, M. López. "Machine Learning for Asset Managers." Cambridge University Press, 2020.
2. Ng, A. "Machine Learning Yearning." deeplearning.ai, 2018.

# E-Commerce Big Data Processing Pipeline

A PySpark-based data pipeline for processing and analyzing e-commerce behavior data.

## What It Does

1. **Loads** e-commerce event data (views, carts, purchases) from CSV
2. **Cleans** the data — handles missing values, removes duplicates, filters invalid records
3. **Transforms** — converts timestamps, extracts date features (month, day of week)
4. **Prepares features** — calculates engagement scores and total spend per user-product pair
5. **Exports** cleaned and feature-engineered data to Parquet and CSV

## Requirements

- Python 3.8+
- Apache Spark
- PySpark

## Setup

```powershell
pip install pyspark
```

## Run

```powershell
python ecommerce_data_pipeline.py
```

## Input

Place a `kaggle_ecommerce_dataset.csv` in the same directory, or the pipeline will generate synthetic demo data automatically.

## Output

Processed data is saved to `processed_data/`:
- `ecommerce_features.parquet` — feature-engineered data (Parquet format)
- `ecommerce_cleaned.csv` — cleaned raw data (CSV format)

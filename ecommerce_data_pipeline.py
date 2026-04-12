# ============================================================
# E-Commerce Big Data Processing & Feature Preparation Pipeline
# ============================================================
# Developed for: Data Handling, Big Data Processing, Feature Prep & Storage
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, to_timestamp, month, dayofweek, udf, sum as spark_sum
from pyspark.sql.types import IntegerType, DoubleType, StringType, StructType, StructField
import os

# ─────────────────────────────────────────────
# 1. Initialize Big Data Processing (Apache Spark)
# ─────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("EcommerceDataPipeline") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Apache Spark Session initialized for Big Data Processing.\n")

# ─────────────────────────────────────────────
# 2. Data Handling: Load Dataset (CSV from Kaggle)
# ─────────────────────────────────────────────
# Assuming a standard Kaggle e-commerce behavior dataset structure
schema = StructType([
    StructField("event_time", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("category_id", StringType(), True),
    StructField("category_code", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("user_session", StringType(), True)
])

# Attempting to load the dataset
DATA_PATH = "kaggle_ecommerce_dataset.csv"  # Target Kaggle dataset
try:
    print(f"📥 Attempting to load dataset from {DATA_PATH}...")
    df = spark.read.csv(DATA_PATH, header=True, schema=schema)
    print(f"📊 Initial record count: {df.count():,}\n")
except Exception as e:
    print(f"⚠️ Dataset {DATA_PATH} not found. Creating a representative dataframe for pipeline demonstration...\n")
    # Synthetic data representative of Kaggle ecommerce dataset
    data = [
        ("2023-10-01 00:00:00 UTC", "view", 1001, "205301", "electronics.smartphone", "apple", 999.99, 501, "sess-1"),
        ("2023-10-01 00:05:00 UTC", "cart", 1001, "205301", "electronics.smartphone", "apple", 999.99, 501, "sess-1"),
        ("2023-10-01 00:10:00 UTC", "purchase", 1001, "205301", "electronics.smartphone", "apple", 999.99, 501, "sess-1"),
        ("2023-10-01 00:15:00 UTC", "view", 1002, "205302", "appliances.kitchen", None, 150.00, 502, "sess-2"), # missing brand
        ("2023-10-01 00:20:00 UTC", "view", 1002, "205302", "appliances.kitchen", None, 150.00, 502, "sess-2"), # duplicate view
        (None, "view", 1003, "205303", None, "samsung", -50.0, 503, "sess-3"), # invalid data: missing time, negative price
    ]
    df = spark.createDataFrame(data, schema)
    print("📊 Sample Data:")
    df.show(truncate=False)

# ─────────────────────────────────────────────
# 3. Data Handling: Handle Missing Values & Clean Data
# ─────────────────────────────────────────────
print("🧹 Cleaning data and handling missing values...")

# Check missing values
print("Missing values per column before cleaning:")
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# a. Drop rows where critical fields are missing (e.g., event_time, user_id, product_id)
cleaned_df = df.dropna(subset=["event_time", "user_id", "product_id"])

# b. Handle missing categoricals (Fill NA for brand and category)
cleaned_df = cleaned_df.fillna({
    "brand": "unknown_brand",
    "category_code": "uncategorized"
})

# c. Clean invalid numerical values (e.g. price must be >= 0)
cleaned_df = cleaned_df.filter(col("price") >= 0)

# d. Remove exact duplicates
cleaned_df = cleaned_df.dropDuplicates()

print(f"📈 Record count after handling missing values and cleaning: {cleaned_df.count():,}\n")


# ─────────────────────────────────────────────
# 4. Big Data Processing: Transformations
# ─────────────────────────────────────────────
print("🔄 Performing big data transformations...")

# Convert event_time to native Spark TimestampType
transformed_df = cleaned_df.withColumn(
    "event_timestamp", to_timestamp(col("event_time"), "yyyy-MM-dd HH:mm:ss 'UTC'")
).drop("event_time")

# Extract structured date parts for deeper analysis
transformed_df = transformed_df.withColumn("interaction_month", month(col("event_timestamp"))) \
                               .withColumn("interaction_dayofweek", dayofweek(col("event_timestamp")))

print("📅 Transformed Structured Data Sample:")
transformed_df.select("user_id", "product_id", "event_type", "event_timestamp", "interaction_dayofweek").show(truncate=False)


# ─────────────────────────────────────────────
# 5. Feature Preparation (For downstream modeling)
# ─────────────────────────────────────────────
print("⚙️ Preparing structured features...")

# Create an implicit interaction weight feature based on event type
def event_weight(event_type):
    if event_type == 'purchase': return 5.0
    elif event_type == 'cart': return 3.0
    elif event_type == 'view': return 1.0
    return 0.0

event_weight_udf = udf(event_weight, DoubleType())

# Add the feature column
feature_df = transformed_df.withColumn("interaction_weight", event_weight_udf(col("event_type")))

# Aggregate features per user-product pairing
final_features_df = feature_df.groupBy("user_id", "product_id") \
    .agg(
        count("event_type").alias("total_interactions"),
        spark_sum("interaction_weight").alias("implicit_engagement_score"),
        spark_sum(when(col("event_type") == "purchase", col("price")).otherwise(0)).alias("total_spent")
    )

print("🎯 Final prepared features ready for algorithms (Sample):")
final_features_df.show(5)


# ─────────────────────────────────────────────
# 6. Data Storage: Save Processed Data
# ─────────────────────────────────────────────
# Organize Datasets in an output directory
OUT_DIR = "c:/Users/A/Documents/ecommerce_system/processed_data"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PARQUET = f"{OUT_DIR}/ecommerce_features.parquet"
OUTPUT_CSV = f"{OUT_DIR}/ecommerce_cleaned.csv"

print("💾 Organizing and storing datasets...")

# Save Engineered Features to Parquet (Highly optimized columnar format for Big Data)
final_features_df.write.mode("overwrite").parquet(OUTPUT_PARQUET)
print(f"✅ Features saved successfully to: {OUTPUT_PARQUET} (Parquet Format)")

# Save Cleaned Data to CSV
# Cast timestamp to string so it can be safely exported to CSV
csv_export_df = transformed_df.withColumn("event_timestamp", col("event_timestamp").cast("string"))
csv_export_df.write.mode("overwrite").option("header", "true").csv(OUTPUT_CSV)
print(f"✅ Cleaned raw data saved to: {OUTPUT_CSV} (CSV Format)\n")

# ─────────────────────────────────────────────
print("🛑 Pipeline execution completed successfully. Stopping Spark session.")
spark.stop()

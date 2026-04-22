#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTTM Random Forest Feature Importance Analysis
PySpark version of rf.scala
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer, IndexToString
from pyspark.sql.types import DoubleType
import sys

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: rf_pyspark.py <job_id> [area]")
        print("  job_id: Input file name (without extension)")
        print("  area: Cluster area (default: local)")
        sys.exit(1)

    job_id = sys.argv[1]
    dt = job_id[0:8]  # First 8 characters as partition value
    area = sys.argv[2] if len(sys.argv) > 2 else "local"

    # Configure SparkSession based on area
    if area == "local":
        spark = SparkSession.builder \
            .appName("TTTM_RandomForest") \
            .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse") \
            .config("hive.metastore.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse") \
            .enableHiveSupport() \
            .getOrCreate()
        input_path = f"/root/{job_id}.csv"
    else:
        # Remote cluster settings would go here
        spark = SparkSession.builder \
            .appName("TTTM_RandomForest") \
            .enableHiveSupport() \
            .getOrCreate()
        input_path = f"/fcbig/{job_id}"

    print(f"=== TTTM Random Forest Analysis ===")
    print(f"Job ID: {job_id}")
    print(f"Partition (dt): {dt}")
    print(f"Area: {area}")
    print(f"Input path: {input_path}")

    # Read input file (CSV format with header)
    if area == "local":
        # Local file - use CSV with comma separator
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .option("sep", ",") \
            .csv(input_path)
    else:
        # HDFS file
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .option("sep", ",") \
            .csv(input_path)

    print("\n=== Input Data Schema ===")
    df.printSchema()

    print("\n=== Input Data Sample (first 5 rows) ===")
    df.show(5, truncate=False)

    # Get column names
    col_name_list = df.columns
    print(f"\n=== Columns ({len(col_name_list)}) ===")
    print(f"Label column: {col_name_list[0]}")
    print(f"Feature columns: {col_name_list[1:]}")

    # Convert all columns to DoubleType
    for col in col_name_list:
        df = df.withColumn(col, df[col].cast(DoubleType()))

    # Assemble features
    feature_cols = col_name_list[1:]  # All columns except Label
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    # Index the label column
    label_indexer = StringIndexer(
        inputCol="Label",
        outputCol="indexedLabel"
    )

    # Index features
    feature_indexer = VectorIndexer(
        inputCol="features",
        outputCol="indexedFeatures",
        maxCategories=4
    )

    # Train a RandomForest model
    rf = RandomForestClassifier(
        labelCol="indexedLabel",
        featuresCol="indexedFeatures",
        maxDepth=30,
        numTrees=500
    )

    # Convert indexed labels back to original labels
    label_converter = IndexToString(
        inputCol="prediction",
        outputCol="predictedLabel",
        labels=label_indexer.fit(df).labels
    )

    # Create pipeline (assembler must come first to create 'features' column)
    pipeline = Pipeline(stages=[assembler, label_indexer, feature_indexer, rf, label_converter])

    # Train model
    print("\n=== Training Random Forest ===")
    print(f"Parameters: maxDepth=30, numTrees=500")
    model = pipeline.fit(df)

    # Extract feature importances (rf is at stages[3] after assembler, label_indexer, feature_indexer)
    rf_model = model.stages[3]
    feature_importances = rf_model.featureImportances.toArray()

    print("\n=== Feature Importances ===")
    print(f"{'Feature':<30} {'Importance':<15}")
    print("-" * 45)

    # Build INSERT statement
    insert_sql = f"INSERT INTO TABLE bizanal.tttm PARTITION(dt='{dt}') VALUES "
    values = []

    for i, importance in enumerate(feature_importances):
        feature_name = col_name_list[i + 1]  # Skip Label column
        print(f"{feature_name:<30} {importance:<15.6f}")
        values.append(f"('{job_id}', '{feature_name}', {importance})")

    # Execute INSERT
    insert_sql += ", ".join(values)

    print(f"\n=== Executing SQL ===")
    spark.sql(insert_sql)

    print(f"\n=== Results saved to bizanal.tttm (dt={dt}) ===")
    print(f"Total features: {len(feature_importances)}")

    spark.stop()

if __name__ == "__main__":
    main()

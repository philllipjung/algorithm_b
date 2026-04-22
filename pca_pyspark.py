#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTTM PCA (Principal Component Analysis)
PySpark version of pca.scala
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array
import sys

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: pca_pyspark.py <job_id> [area]")
        print("  job_id: Input file name (without extension)")
        print("  area: Cluster area (default: local)")
        sys.exit(1)

    job_id = sys.argv[1]
    area = sys.argv[2] if len(sys.argv) > 2 else "local"

    # Configure SparkSession based on area
    if area == "local":
        spark = SparkSession.builder \
            .appName("TTTM_PCA") \
            .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse") \
            .enableHiveSupport() \
            .getOrCreate()
        input_path = f"/root/{job_id}.csv"
        output_path = f"/tmp/pca_output_{job_id}"
    else:
        # Remote cluster settings (ichbig)
        spark = SparkSession.builder \
            .appName("TTTM_PCA") \
            .config("spark.sql.warehouse.dir", "/fcbig/warehouse") \
            .config("hive.metastore.uris", "thrift://fcbig-06-12:9083") \
            .enableHiveSupport() \
            .getOrCreate()
        input_path = f"/fcbig/pca/{job_id}"
        output_path = f"/fcbig/output/{job_id}"

    print(f"=== TTTM PCA Analysis ===")
    print(f"Job ID: {job_id}")
    print(f"Area: {area}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    # Read CSV file with null handling
    df = spark.read \
        .option("header", "false") \
        .option("inferSchema", "true") \
        .csv(input_path) \
        .na.drop("any")

    print(f"\n=== Input Data ===")
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    # Get column names (exclude _c0 which is the ID column)
    col_names = [c for c in df.columns if c != "_c0"]
    print(f"Feature columns: {len(col_names)}")

    # Assemble features
    assembler = VectorAssembler(
        inputCols=col_names,
        outputCol="features",
        handleInvalid="skip"
    )

    # Transform to create features column
    assembled_df = assembler.transform(df)

    # Standardize features
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )

    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)

    # Apply PCA with K=2
    pca = PCA(
        inputCol="scaledFeatures",
        outputCol="pcaFeatures",
        k=2
    )

    pca_model = pca.fit(scaled_df)
    pca_result = pca_model.transform(scaled_df)

    # Print explained variance
    explained_variance = pca_model.explainedVariance.toArray()
    print(f"\n=== PCA Results ===")
    print(f"Explained Variance Ratio:")
    print(f"  PC1: {explained_variance[0]:.4f} ({explained_variance[0]*100:.2f}%)")
    print(f"  PC2: {explained_variance[1]:.4f} ({explained_variance[1]*100:.2f}%)")
    print(f"  Total: {explained_variance.sum():.4f} ({explained_variance.sum()*100:.2f}%)")

    # Extract PCA features and convert to separate columns
    # Convert Vector to Array using Spark SQL built-in function
    result_df = pca_result.select(
        col("_c0").alias("id"),
        vector_to_array("pcaFeatures")[0].alias("pc1"),
        vector_to_array("pcaFeatures")[1].alias("pc2")
    )

    # Show sample results
    print(f"\n=== Sample Results (first 10 rows) ===")
    result_df.show(10, truncate=False)

    # Save results as CSV
    print(f"\n=== Saving Results ===")
    result_df.coalesce(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(output_path)

    print(f"Results saved to: {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()

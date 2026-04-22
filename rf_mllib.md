# Spark MLlib Random Forest Feature Importance Analysis

## 개요

PySpark MLlib를 사용한 Random Forest Feature Importance 분석 스크립트입니다.
TTTM(Test/Train Parameters Management) 데이터를 분석하여 변수 중요도를 계산하고
Hive 테이블에 저장합니다.

## 설치된 구성 요소

| 컴포넌트 | 버전 | 설치 경로 |
|----------|------|----------|
| Java | 1.8.0_482 | /usr/lib/jvm/java-8-openjdk-amd64 |
| Hadoop | 3.3.6 | /opt/hadoop |
| Spark | 3.5.3 | /opt/spark |
| Hive | 3.1.3 | /opt/hive |
| Conda 환경 | Python 3.9.25 | /opt/anaconda3/envs/hynix |
| numpy | 1.26.4 | hynix 환경 |

## 스크립트 정보

**파일명**: `/root/rf_pyspark.py`

**입력 포맷**: CSV (Comma-Separated Values)

**출력**: Hive 테이블 `bizanal.tttm`

## 입력 파일 형식

### CSV 파일 구조

```csv
Label,temperature,pressure,velocity,flow_rate,density,humidity,voltage,current,frequency,amplitude,resistance,capacitance,inductance,power,energy,efficiency,loss,gain,time_constant,impedance
0,1.2,3.4,5.6,7.8,9.1,2.3,4.5,6.7,8.9,1.1,2.4,3.7,5.2,7.1,9.4,2.6,4.8,6.9,8.2,2.0
0,1.5,3.1,5.2,7.5,8.8,2.1,4.8,6.3,8.5,1.3,2.6,3.9,5.4,7.3,9.1,2.8,4.5,6.6,8.7,1.5
```

### 컬럼 설명

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| Label | int | 타겟 변수 (0 또는 1) |
| temperature | double | 온도 |
| pressure | double | 압력 |
| velocity | double | 속도 |
| flow_rate | double | 유량 |
| density | double | 밀도 |
| humidity | double | 습도 |
| voltage | double | 전압 |
| current | double | 전류 |
| frequency | double | 주파수 |
| amplitude | double | 진폭 |
| resistance | double | 저항 |
| capacitance | double | 커패시턴스 |
| inductance | double | 인덕턴스 |
| power | double | 전력 |
| energy | double | 에너지 |
| efficiency | double | 효율 |
| loss | double | 손실 |
| gain | double | 이득 |
| time_constant | double | 시간 상수 |
| impedance | double | 임피던스 |

## 사용 방법

### 1. 실행 환경 설정

```bash
# Java 8 설정
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Conda 환경 설정 (numpy 필요)
export PYSPARK_PYTHON=/opt/anaconda3/envs/hynix/bin/python
export PYSPARK_DRIVER_PYTHON=/opt/anaconda3/envs/hynix/bin/python
```

### 2. HDFS 시작

```bash
# HDFS 시작
/opt/hadoop/sbin/start-dfs.sh

# YARN 시작
/opt/hadoop/sbin/start-yarn.sh

# 확인
jps
```

### 3. 스크립트 실행

```bash
/opt/spark/bin/spark-submit --master local[4] /root/rf_pyspark.py rf_test_data local
```

### 4. 실행 인자

| 인자 | 설명 | 예시 |
|------|------|------|
| `job_id` | 작업 ID (입력 파일명) | `rf_test_data` |
| `area` | 클러스터 영역 | `local`, `ich`, `wxh` |

## Spark MLlib 컴포넌트

### 사용된 MLlib 라이브러리

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer, IndexToString
```

### ML Pipeline 구조

```
┌─────────────────────────────────────────────────────────┐
│                    Pipeline                             │
├─────────────────────────────────────────────────────────┤
│ 1. VectorAssembler                                      │
│    - 개별 feature 컬럼을 하나의 features 벡터로 조립    │
│    - inputCols: [temperature, pressure, ..., impedance] │
│    - outputCol: "features"                              │
├─────────────────────────────────────────────────────────┤
│ 2. StringIndexer                                        │
│    - Label 컬럼을 수치형으로 변환                        │
│    - inputCol: "Label"                                  │
│    - outputCol: "indexedLabel"                          │
├─────────────────────────────────────────────────────────┤
│ 3. VectorIndexer                                        │
│    - features 벡터의 카테고리형 컬럼을 인덱싱           │
│    - inputCol: "features"                               │
│    - outputCol: "indexedFeatures"                       │
│    - maxCategories: 4                                   │
├─────────────────────────────────────────────────────────┤
│ 4. RandomForestClassifier                               │
│    - Random Forest 분류 모델 학습                       │
│    - labelCol: "indexedLabel"                           │
│    - featuresCol: "indexedFeatures"                     │
│    - maxDepth: 30                                       │
│    - numTrees: 500                                      │
├─────────────────────────────────────────────────────────┤
│ 5. IndexToString                                        │
│    - 예측 결과를 원래 라벨로 변환                      │
│    - inputCol: "prediction"                             │
│    - outputCol: "predictedLabel"                        │
└─────────────────────────────────────────────────────────┘
```

### Random Forest 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| maxDepth | 30 | 트리의 최대 깊이 |
| numTrees | 500 | 생성할 트리 개수 |
| labelCol | indexedLabel | 라벨 컬럼 |
| featuresCol | indexedFeatures | 피처 벡터 컬럼 |

## 실행 예시

### 테스트 실행 결과

```bash
$ /opt/spark/bin/spark-submit --master local[4] /root/rf_pyspark.py rf_test_data local

=== TTTM Random Forest Analysis ===
Job ID: rf_test_data
Partition (dt): rf_test_
Area: local
Input path: /root/rf_test_data.csv

=== Input Data Schema ===
root
 |-- Label: double (nullable = true)
 |-- temperature: double (nullable = true)
 |-- pressure: double (nullable = true)
 ...
 |-- impedance: double (nullable = true)

=== Columns (21) ===
Label column: Label
Feature columns: ['temperature', 'pressure', 'velocity', ...]

=== Training Random Forest ===
Parameters: maxDepth=30, numTrees=500

=== Feature Importances ===
Feature                        Importance
---------------------------------------------
temperature                    0.336387
humidity                       0.194300
pressure                       0.134202
resistance                     0.110252
amplitude                      0.064908
voltage                        0.058938
velocity                       0.032283
current                        0.012609
frequency                      0.011854
inductance                     0.008639
gain                           0.007216
energy                         0.006978
capacitance                    0.004522
efficiency                     0.003236
loss                           0.003174
time_constant                  0.003156
power                          0.002883
flow_rate                      0.002991
density                        0.001192
impedance                      0.000279

=== Executing SQL ===
=== Results saved to bizanal.tttm (dt=rf_test_) ===
Total features: 20
```

## 결과 저장

### Hive 테이블 구조

```sql
-- bizanal.tttm 테이블
CREATE TABLE bizanal.tttm (
    job_id STRING,
    param STRING,
    value DOUBLE
)
PARTITIONED BY (dt STRING)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES (
    'field.delim'=',',
    'serialization.format'=','
)
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/bizanal.tttm';
```

### 결과 조회

```bash
# Hive CLI 시작
/opt/hive/bin/hive

# bizanal 데이터베이스 사용
USE bizanal;

# 결과 조회 (중요도 순)
SELECT param, value
FROM tttm
WHERE dt='rf_test_'
ORDER BY CAST(value AS DOUBLE) DESC;

# 전체 결과 확인
SELECT * FROM tttm WHERE dt='rf_test_';
```

## Feature Importance 해석

### 상위 10개 중요 변수

| 순위 | Feature | Importance | 설명 |
|-----|---------|------------|------|
| 1 | temperature | 0.3364 (33.6%) | 가장 중요한 변수 |
| 2 | humidity | 0.1943 (19.4%) | 2번째 중요 |
| 3 | pressure | 0.1342 (13.4%) | 3번째 중요 |
| 4 | resistance | 0.1103 (11.0%) | 4번째 중요 |
| 5 | amplitude | 0.0649 (6.5%) | 5번째 중요 |
| 6 | voltage | 0.0589 (5.9%) | 6번째 중요 |
| 7 | velocity | 0.0323 (3.2%) | 7번째 중요 |
| 8 | current | 0.0126 (1.3%) | 8번째 중요 |
| 9 | frequency | 0.0119 (1.2%) | 9번째 중요 |
| 10 | inductance | 0.0086 (0.9%) | 10번째 중요 |

### 중요도 합계

- 상위 3개 변수 (temperature, humidity, pressure): 66.5%
- 상위 5개 변수: 84.0%
- 상위 10개 변수: 95.6%

## 스크립트 소스 코드

### 전체 소스

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTTM Random Forest Feature Importance Analysis
PySpark version using Spark MLlib
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
```

## 트러블슈팅

### 문제 1: numpy 모듈 없음

**에러 메시지**:
```
ModuleNotFoundError: No module named 'numpy'
```

**해결**:
```bash
export PYSPARK_PYTHON=/opt/anaconda3/envs/hynix/bin/python
export PYSPARK_DRIVER_PYTHON=/opt/anaconda3/envs/hynix/bin/python
```

### 문제 2: HDFS 연결 거부

**에러 메시지**:
```
Call From ... to localhost:9000 failed on connection exception
```

**해결**:
```bash
/opt/hadoop/sbin/start-dfs.sh
/opt/hadoop/sbin/start-yarn.sh
jps  # NameNode, DataNode 확인
```

### 문제 3: 데이터에 null 값

**에러 메시지**:
```
Encountered null while assembling a row with handleInvalid = "error"
```

**해결**: 데이터 파일에 null 값이 없도록 모든 행에 21개 컬럼이 있는지 확인

### 문제 4: VectorAssembler 'features' 없음

**에러 메시지**:
```
features does not exist. Available: Label, temperature, ...
```

**해결**: Pipeline에 assembler가 포함되어 있는지 확인
```python
pipeline = Pipeline(stages=[assembler, label_indexer, feature_indexer, rf, label_converter])
```

### 문제 5: Stage 인덱스 오류

**에러 메시지**:
```
AttributeError: 'VectorIndexerModel' object has no attribute 'featureImportances'
```

**해결**: RandomForest 모델은 `stages[3]`에 있습니다
```python
rf_model = model.stages[3]  # assembler, label_indexer, feature_indexer, rf
```

## Spark MLlib vs RDD API

### MLlib (DataFrame-based API) - 현재 사용

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

rf = RandomForestClassifier(labelCol="indexedLabel",
                           featuresCol="indexedFeatures",
                           maxDepth=30, numTrees=500)
```

**장점**:
- 성능 최적화됨
- Pipeline 사용 가능
- DataFrame과 통합
- 더 나은 API

### RDD-based API (구버전)

```python
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint

model = RandomForest.trainClassifier(data, numClasses=2,
                                     categoricalFeaturesInfo={},
                                     numTrees=500,
                                     featureSubsetStrategy="auto",
                                     impurity='gini',
                                     maxDepth=30,
                                     maxBins=32)
```

**단점**:
- 성능이 낮음
- Pipeline 사용 불가
- RDD 직접 다뤄야 함
- Deprecated 됨

## 다음 단계

1. 하이퍼파라미터 튜닝 (maxDepth, numTrees)
2. Cross-validation 추가
3. 다른 분류 알고리즘 비교 (Gradient-Boosted Trees, Logistic Regression)
4. Feature Selection 자동화
5. 모델 성능 평가 지표 추가 (Accuracy, Precision, Recall, F1-Score)
6. 모델 저장 및 로드 기능 추가

## 참고 자료

| 리소스 | URL |
|--------|-----|
| Spark MLlib Guide | https://spark.apache.org/docs/latest/ml-guide.html |
| RandomForestClassifier | https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html |
| Spark ML Pipelines | https://spark.apache.org/docs/latest/ml-pipeline.html |
| Hive Language Manual | https://cwiki.apache.org/confluence/display/Hive/LanguageManual |

---

**작성일**: 2026-04-17
**버전**: 1.0
**Python**: 3.9.25
**Spark**: 3.5.3
**Spark MLlib**: 3.5.3

# Spark MLlib PCA (Principal Component Analysis)

## 개요

PySpark MLlib를 사용한 PCA (주성분 분석) 스크립트입니다.
다차원 데이터를 2차원으로 축소하여 데이터의 패턴을 시각화하고 분석합니다.

## 설치된 구성 요소

| 컴포넌트 | 버전 | 설치 경로 |
|----------|------|----------|
| Java | 1.8.0_482 | /usr/lib/jvm/java-8-openjdk-amd64 |
| Hadoop | 3.3.6 | /opt/hadoop |
| Spark | 3.5.3 | /opt/spark |
| Conda 환경 | Python 3.9.25 | /opt/anaconda3/envs/hynix |
| numpy | 1.26.4 | hynix 환경 |

## 스크립트 정보

**파일명**: `/root/pca_pyspark.py`

**입력 포맷**: CSV (Comma-Separated Values)

**출력**: CSV 파일 (ID, PC1, PC2)

## PCA란 무엇인가?

### 주성분 분석 (Principal Component Analysis)

PCA는 다차원 데이터를 저차원 공간으로 변환하는 차원 축소 기법입니다.

**목적:**
- 데이터의 분산을 최대한 보존하면서 차원 축소
- 상관관계가 높은 변수들을 독립적인 주성분으로 변환
- 데이터 시각화 및 노이즈 제거

**작동 원리:**
1. 데이터 표준화 (StandardScaler)
2. 공분산 행렬 계산
3. 고유값 분해 (Eigenvalue Decomposition)
4. 상위 k개의 주성분 선택

## 입력 파일 형식

### CSV 파일 구조

```csv
0,1.2,3.4,5.6,7.8,9.1,2.3,4.5,6.7,8.9,1.1,2.4,3.7,5.2,7.1,9.4,2.6,4.8,6.9,8.2,1.5
1,1.5,3.1,5.2,7.5,8.8,2.1,4.8,6.3,8.5,1.3,2.6,3.9,5.4,7.3,9.1,2.8,4.5,6.6,8.7,1.6
2,2.1,4.2,6.3,8.1,9.5,3.2,5.1,7.2,9.3,2.4,3.5,4.8,6.1,8.3,9.7,3.7,5.6,7.8,9.1,1.7
```

### 컬럼 구조

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| _c0 | int | ID (식별자) |
| _c1 ~ _c20 | double | Feature 변수 (20개) |

## 사용 방법

### 1. 실행 환경 설정

```bash
# Java 8 설정
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Conda 환경 설정
export PYSPARK_PYTHON=/opt/anaconda3/envs/hynix/bin/python
export PYSPARK_DRIVER_PYTHON=/opt/anaconda3/envs/hynix/bin/python
```

### 2. 스크립트 실행

```bash
# 로컬 실행
/opt/spark/bin/spark-submit --master local[4] /root/pca_pyspark.py pca_test_data local

# 원격 클러스터 실행
/opt/spark/bin/spark-submit --master local[4] /root/pca_pyspark.py <job_id> ich
```

### 3. 실행 인자

| 인자 | 설명 | 예시 |
|------|------|------|
| `job_id` | 작업 ID (입력 파일명) | `pca_test_data` |
| `area` | 클러스터 영역 | `local`, `ich` |

## Spark MLlib 컴포넌트

### 사용된 MLlib 라이브러리

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array
```

### ML Pipeline 구조

```
┌─────────────────────────────────────────────────────────┐
│                    PCA Pipeline                         │
├─────────────────────────────────────────────────────────┤
│ 1. VectorAssembler                                      │
│    - 개별 feature 컬럼을 하나의 features 벡터로 조립    │
│    - inputCols: [_c1, _c2, ..., _c20]                  │
│    - outputCol: "features"                              │
│    - handleInvalid: "skip" (null 값 건너뛰기)           │
├─────────────────────────────────────────────────────────┤
│ 2. StandardScaler                                        │
│    - 피처 표준화 (평균 0, 표준편차 1)                   │
│    - inputCol: "features"                               │
│    - outputCol: "scaledFeatures"                        │
│    - withMean: true (평균 중앙화)                       │
│    - withStd: true (표준편차 스케일링)                  │
├─────────────────────────────────────────────────────────┤
│ 3. PCA                                                   │
│    - 주성분 분석 수행                                   │
│    - inputCol: "scaledFeatures"                         │
│    - outputCol: "pcaFeatures"                           │
│    - k: 2 (주성분 개수)                                 │
├─────────────────────────────────────────────────────────┤
│ 4. vector_to_array (Spark MLlib 내장 함수)             │
│    - PCA 벡터를 개별 컬럼으로 분리                      │
│    - pc1: 제1주성분                                     │
│    - pc2: 제2주성분                                     │
└─────────────────────────────────────────────────────────┘
```

### PCA 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| inputCol | scaledFeatures | 입력 컬럼 (표준화된 피처) |
| outputCol | pcaFeatures | 출력 컬럼 (PCA 변환 결과) |
| k | 2 | 생성할 주성분 개수 |

### StandardScaler 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| withMean | true | 평균을 0으로 중앙화 |
| withStd | true | 단위 분산으로 스케일링 |

## 실행 예시

### 테스트 실행 결과

```bash
$ /opt/spark/bin/spark-submit --master local[4] /root/pca_pyspark.py pca_test_data local

=== TTTM PCA Analysis ===
Job ID: pca_test_data
Area: local
Input path: /root/pca_test_data.csv
Output path: /tmp/pca_output_pca_test_data

=== Input Data ===
Rows: 50
Columns: 21
Feature columns: 20

=== PCA Results ===
Explained Variance Ratio:
  PC1: 0.3376 (33.76%)
  PC2: 0.2615 (26.15%)
  Total: 0.5992 (59.92%)

=== Sample Results (first 10 rows) ===
+---+-------------------+-------------------+
|id |pc1                |pc2                |
+---+-------------------+-------------------+
|0  |1.7125133890855853 |-1.9164425440396309|
|1  |2.1881278132002744 |-1.355549552882934 |
|2  |-1.1582460658779024|-2.5894676126173986|
|3  |1.2504873259478668 |-1.860655885991348 |
|4  |-3.4455042848543633|-3.4371248951627846|
|5  |2.996823535564401  |-0.8744147930360358|
|6  |1.803783584289798  |-1.509991819075541 |
|7  |-4.3087308128596415|-3.240221951376838 |
|8  |0.8494766478344951 |-1.7463355639520008|
|9  |-2.223507054447515 |-2.6379121321086036|
+---+-------------------+-------------------+

=== Saving Results ===
Results saved to: /tmp/pca_output_pca_test_data
```

## 결과 해석

### 설명 분산 (Explained Variance)

주성분이 데이터의 분산을 얼마나 설명하는지 나타내는 지표입니다.

```
PC1: 33.76% - 첫 번째 주성분이 데이터 분산의 33.76% 설명
PC2: 26.15% - 두 번째 주성분이 데이터 분산의 26.15% 설명
Total: 59.92% - 두 주성분이 전체 분산의 59.92% 설명
```

### PCA 결과 구조

```csv
id,pc1,pc2
0,1.7125133890855853,-1.9164425440396309
1,2.1881278132002744,-1.355549552882934
...
```

| 컬럼명 | 설명 |
|--------|------|
| id | 원본 데이터의 식별자 (_c0) |
| pc1 | 제1주성분 값 (가장 큰 분산 설명) |
| pc2 | 제2주성분 값 (두 번째로 큰 분산 설명) |

## PCA 결과 시각화 예시

### 2D 산점도

```
    PC2
     ↑
  2  |         ●
  1  |    ●        ●
  0  |              ●
 -1  |    ●     ●     ●
 -2  |       ●  ●  ●
 -3  |  ●           ●
     +--------------------------→ PC1
       -4  -2  0  2  4
```

### 데이터 군집 분석

- **우측 상단** (PC1 > 0, PC2 > 0): 특정 패턴을 가진 데이터 그룹
- **좌측 하단** (PC1 < 0, PC2 < 0): 다른 패턴을 가진 데이터 그룹

## 스크립트 소스 코드

### 전체 소스

```python
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
```

## Scala vs PySpark 비교

### Scala 코드 (pca.scala)

```scala
// VectorAssembler
val assembler = new VectorAssembler()
  .setInputCols(colNames.toArray)
  .setOutputCol("features")

// StandardScaler
val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

// PCA
val pca = new PCA()
  .setInputCol("scaledFeatures")
  .setOutputCol("pcaFeatures")
  .setK(2)
  .fit(scalerDF)

// 결과 변환
val result = pca.transform(scalerDF).select("_c0", "pcaFeatures")
  .map(row => {
    val id = row.getAs[Int](0)
    val features = row.getAs[org.apache.spark.ml.linalg.DenseVector](1).toArray
    (id, features(0).toString(), features(1).toString)
  })
```

### PySpark 코드 (pca_pyspark.py)

```python
# VectorAssembler
assembler = VectorAssembler(
    inputCols=col_names,
    outputCol="features",
    handleInvalid="skip"
)

# StandardScaler
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True
)

# PCA
pca = PCA(
    inputCol="scaledFeatures",
    outputCol="pcaFeatures",
    k=2
)

# 결과 변환 (vector_to_array 사용)
result_df = pca_result.select(
    col("_c0").alias("id"),
    vector_to_array("pcaFeatures")[0].alias("pc1"),
    vector_to_array("pcaFeatures")[1].alias("pc2")
)
```

## vector_to_array vs UDF 비교

### vector_to_array (권장) - Spark 3.0+

```python
from pyspark.ml.functions import vector_to_array

result_df = pca_result.select(
    col("_c0").alias("id"),
    vector_to_array("pcaFeatures")[0].alias("pc1"),
    vector_to_array("pcaFeatures")[1].alias("pc2")
)
```

**장점:**
- Spark SQL 내장 함수
- 직렬화 불필요
- UDF보다 빠름 (Python ↔ JVM 변환 없음)
- Catalyst 옵티마이저 최적화 가능

### UDF (구버전 호환용)

```python
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import DoubleType

def get_element(vector, index):
    return float(vector[index])

udf_get_element = udf(get_element, DoubleType())

result_df = pca_result.select(
    col("_c0").alias("id"),
    udf_get_element("pcaFeatures", lit(0)).alias("pc1"),
    udf_get_element("pcaFeatures", lit(1)).alias("pc2")
)
```

**단점:**
- Python 함수 직렬화 필요
- JVM ↔ Python 변환 오버헤드
- Catalyst 최적화 불가

## 트러블슈팅

### 문제 1: vector_to_array 함수 import 경로

**올바른 import 경로**:
```python
# 올바름 (Spark 3.0+)
from pyspark.ml.functions import vector_to_array

# 틀림 (pyspark.sql.functions에 없음)
# from pyspark.sql.functions import vector_to_array
```

**UDF 대신 vector_to_array 사용 장점**:
- Spark SQL 내장 함수로 직렬화 불필요
- UDF보다 성능 우수 (Python ↔ JVM 변환 없음)
- Catalyst 옵티마이저가 최적화 가능

### 문제 2: PCA 결과 해석

**질문**: 설명 분산이 낮은 경우 (예: < 60%)

**해결**:
- k 값을 증가 (더 많은 주성분 사용)
- 데이터 전처리 검토 (이상값 제거, 결측값 처리)
- Feature 엔지니어링 (불필요한 변수 제거)

### 문제 3: 메모리 부족

**에러 메시지**:
```
java.lang.OutOfMemoryError: Java heap space
```

**해결**:
```bash
# Spark 메모리 증가
/opt/spark/bin/spark-submit \
  --driver-memory 4g \
  --executor-memory 4g \
  --master local[4] \
  /root/pca_pyspark.py pca_test_data local
```

## PCA 응용 분야

### 1. 차원 축소
- 고차원 데이터를 2~3차원으로 축소
- 시각화 및 탐색적 데이터 분석

### 2. 노이즈 제거
- 상위 주성분만 유지하여 노이즈 제거
- 데이터 압축

### 3. feature 추출
- 상관관계가 높은 변수들을 독립적인 주성분으로 변환
- 머신러닝 모델의 입력으로 사용

### 4. 이상 탐지
- 주성분 공간에서의 거리를 활용한 이상치 탐지
- 재구성 오차(Reconstruction Error) 활용

## PCA와 Random Forest 비교

| 특징 | PCA | Random Forest |
|------|-----|---------------|
| 목적 | 차원 축소 | 분류/회귀 |
| 지도 학습 | 비지도 | 지도 |
| 출력 | 주성분 (변환된 피처) | 예측값, Feature Importance |
| 주요 파라미터 | k (주성분 개수) | maxDepth, numTrees |
| 결과 해석 | 설명 분산 | Feature Importance |
| 시각화 | 가능 (2D/3D) | 어려움 |

## 다음 단계

1. **K 값 최적화**: scree plot을 활용한 최적 k 값 선택
2. **3D PCA**: k=3으로 설정하여 3차원 시각화
3. **군집 분석**: PCA 결과에 K-Means 적용
4. **이상 탐지**: 재구성 오차를 활용한 이상치 탐지
5. **t-SNE/UMAP**: 비선형 차원 축소 기법과 비교

## 참고 자료

| 리소스 | URL |
|--------|-----|
| Spark MLlib PCA | https://spark.apache.org/docs/latest/ml-features.html#pca |
| Spark StandardScaler | https://spark.apache.org/docs/latest/ml-features.html#standardscaler |
| VectorAssembler | https://spark.apache.org/docs/latest/ml-features.html#vectorassembler |
| PCA Tutorial | https://en.wikipedia.org/wiki/Principal_component_analysis |

---

**작성일**: 2026-04-17
**버전**: 1.1 (vector_to_array 사용)
**Python**: 3.9.25
**Spark**: 3.5.3
**Spark MLlib**: 3.5.3

import datetime
import os
import statistics
import sys
import time
import warnings
import gc

import numpy as np
import pandas as pd
import scipy.stats as stats
from minepy import MINE
from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import check_X_y
from joblib import Parallel, delayed

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
from pyhive import presto
from pyhive import hive
from pyspark.sql import SparkSession
from hdfs import InsecureClient

def create_insert_value(*str_values):
    str_query = " ("
    for idx, value in enumerate(str_values):
        if idx == len(str_values)-1:
            str_query += "'" + str(value) + "')\n"
        else:
            str_query += "'" + str(value) + "',"
    return str_query


def filter_timestamp_columns(df):
    filtered_columns = []
    if isinstance(df, pd.DataFrame):
        for k, v in df.loc['mean'].to_dict().items():
            try:
                if len(str(round(v))) == 10:
                    filtered_columns.append(k)
            except ValueError:
                continue
    return filtered_columns

def get_dynamic_clustering_params(n_samples, n_features):
    """
    데이터 크기에 따른 동적 파라미터 튜닝

    Args:
        n_samples: 샘플 수
        n_features: feature 수

    Returns:
        dict: 동적 파라미터
    """
    params = {}

    # PCA n_components
    if n_samples < 100:
        params['n_components'] = 2
    elif n_samples < 1000:
        params['n_components'] = min(5, n_features)
    elif n_samples < 10000:
        params['n_components'] = min(10, n_features // 10)
    else:  # 만 건 이상
        params['n_components'] = min(20, n_features // 20)

    # max_cluster
    if n_samples < 100:
        params['max_cluster'] = min(10, n_samples // 2)
    elif n_samples < 500:
        params['max_cluster'] = min(20, n_samples // 10)
    elif n_samples < 2000:
        params['max_cluster'] = min(30, n_samples // 20)
    else:
        params['max_cluster'] = 30

    # iterations
    if n_samples < 100:
        params['iterations'] = 5
    elif n_samples < 500:
        params['iterations'] = 10
    else:
        params['iterations'] = 5

    # RF iterations
    if n_samples < 100:
        params['rf_iterations'] = 20
    elif n_samples < 500:
        params['rf_iterations'] = 10
    elif n_samples < 5000:
        params['rf_iterations'] = 5
    else:  # 만 건 이상
        params['rf_iterations'] = 3

    # Pre-screening top_n
    if n_samples < 1000:
        params['rf_top_n'] = 100
        params['post_logistic_top_n'] = None  # 소규모에서는 미적용
    elif n_samples < 5000:
        params['rf_top_n'] = 150
        params['post_logistic_top_n'] = 75
    else:  # 만 건 이상
        params['rf_top_n'] = 200
        params['post_logistic_top_n'] = 100

    return params


def get_dynamic_kfold_params(n_samples):
    """
    데이터 크기에 따른 K-Fold 파라미터 튜닝

    Args:
        n_samples: 샘플 수

    Returns:
        dict: {'k': fold 수, 'tol': 정확도 허용 오차}
    """
    if n_samples < 50:
        return {'k': 3, 'tol': 0.0001}
    elif n_samples < 200:
        return {'k': 5, 'tol': 0.0001}
    elif n_samples < 1000:
        return {'k': 10, 'tol': 0.0005}
    else:  # 만 건 이상
        return {'k': 5, 'tol': 0.001}


def profile_dataset(data:pd.DataFrame):
    n_of_rows, n_of_cols = data.shape
    ratio = n_of_cols/n_of_rows
    group_key_in_dataset = False
    group_size = 0
    if 'GROUP_ID' in data.columns.tolist():
        group_size = data['GROUP_ID'].nunique()
        group_key_in_dataset = True
    return group_key_in_dataset, group_size, ratio

def get_clustering_df(data: pd.DataFrame, columns: list, end_time_position: int):
    clustering_df = data[columns].copy()
    end_time_value = clustering_df.iloc[:, end_time_position]
    dt_end_time_value = pd.to_datetime(end_time_value)
    numeric_end_time_value = pd.to_numeric(dt_end_time_value)
    clustering_df = clustering_df.assign(**{
        'Time': dt_end_time_value,
        'Time1': numeric_end_time_value
    })
    clustering_df.reset_index(inplace=True)
    return clustering_df

def get_optimal_cluster_number(data: pd.DataFrame,
                               iterations: int, max_cluster: int, verbose: bool):
    result = []
    cluster_range = range(3, max_cluster)
    for i in np.arange(iterations):
        sum_of_squared_distances = []
        silhouette_avg = []
        for k in cluster_range:
            # MiniBatchKMeans로 변경 (성능 최적화)
            km = MiniBatchKMeans(
                n_clusters=k,
                batch_size=min(100, len(data)),
                random_state=np.random.randint(0, 999999),
                n_init=3,
                max_iter=300
            )
            km = km.fit(data)
            sum_of_squared_distances.append(km.inertia_)
            cluster_labels = km.predict(data)  # 중요: fit 후 predict 호출
            score = silhouette_score(data, cluster_labels)
            silhouette_avg.append(score)
            sum_of_squared_distances.append(km.inertia_)
        cluster_result = [list(cluster_range), silhouette_avg]
        cluster_result = pd.DataFrame(cluster_result)
        cluster_result = pd.DataFrame.transpose(cluster_result)
        max_row = cluster_result.loc[:, 1].idxmax()
        optimal_cluster_number = int(cluster_result.loc[max_row, 0])
        result.append(optimal_cluster_number)
        if verbose:
            print("Optimal_Cluster_Number on "
                  "iteration {} : {}" .format(i, optimal_cluster_number))
    return result

def evaluate_feature_addition_clustering(column, included, X, y):
    """병렬 평가용 헬퍼 함수 (clustering용)"""
    train_features = included + [column]
    train_x = X[train_features]
    model = LogisticRegression(
        penalty='l1',
        multi_class='multinomial',
        solver='saga',
        tol=0.0005,
        random_state=np.random.randint(0, 999999)
    )
    model.fit(train_x, y.values.ravel())
    score = model.score(train_x, y.values.ravel())
    return column, score

def step_forward_k_fold(X, y, k=5, accuracy_update_tol=0.0001, verbose=True):
    kf = StratifiedKFold(n_splits=k)
    kf.get_n_splits(X)
    included = list()
    total_res = pd.DataFrame(
        columns=["# of var", "model", "var_included", "coefs", "Accuracy", "K-Fold Accuracy"])
        
    curr_best_accuracy = -np.inf
    curr_best_kfold_accuracy = -np.inf
    j = 0

    while True:
        changed = False
        included.sort()
        excluded = list(set(X.columns) - set(included))
        excluded.sort()
        if len(excluded) == 0:
            print("All variables are included. Stopping Forward Selection.")
            break

        # 병렬 평가: 모든 excluded feature를 동시에 평가
        if verbose:
            print(f"[Step {len(included)+1}] Evaluating {len(excluded)} features in parallel...")
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_feature_addition_clustering)(col, included, X, y) for col in excluded
        )
        new_accuracy_dict = {col: score for col, score in results}
        best_feature, best_score = max(results, key=lambda x: x[1])

        if verbose:
            print(f"  Best feature: {best_feature} (Accuracy={best_score:.6f})")

        delta = best_score - curr_best_accuracy

        if delta > accuracy_update_tol:
            included.append(best_feature)
            curr_best_accuracy = best_score
            changed = True

        cv_results = pd.DataFrame(columns=["k-fold", "Accuracy"])
        i = 0

        try:
            if len(X) < 30:
                model = LogisticRegression(penalty='l1',
                                           multi_class='multinomial',
                                           solver='saga',
                                           fit_intercept=False,
                                           tol=0.0005, random_state=42)
                model.fit(X[included + [best_feature]], y.values.ravel())
                cv_accuracy = model.score(X[included + [best_feature]], y.values.ravel())
                cv_res = pd.DataFrame({"k_fold": [i], "Accuracy": [cv_accuracy]})
                cv_results = pd.concat([cv_results, cv_res])
            else:
                for _, test_index in kf.split(X, y):
                    X_test = X.iloc[test_index,]
                    y_test = y.iloc[test_index,]
                    model = LogisticRegression(penalty='l1',
                                                multi_class='multinomial',
                                                solver='saga',
                                                max_iter=10000,
                                                fit_intercept=False,
                                                tol=0.0005, random_state=42)
                    model.fit(X_test[included + [best_feature]], y_test.values.ravel())
                    cv_accuracy = model.score(X_test[included + [best_feature]], y_test.values.ravel())
                    cv_res = pd.DataFrame({"k_fold": [i], "Accuracy" : [cv_accuracy]})
                    cv_results = pd.concat([cv_results, cv_res])
                    i += 1
        except:
            model = LogisticRegression(penalty='l1',
                                        multi_class='multinomial',
                                        solver='saga',
                                        max_iter=10000,
                                        fit_intercept=False,
                                        tol=0.0005, random_state=42)
            model.fit(X[included + [best_feature]], y.values.ravel())
            cv_accuracy = model.score(X[included + [best_feature]], y.values.ravel())
            cv_res = pd.DataFrame({"k_fold": [i], "Accuracy": [cv_accuracy]})
            cv_results = pd.concat([cv_results, cv_res])

        kf_accuracy = cv_results['Accuracy'].mean()
        curr_model = LogisticRegression(penalty='l1',
                                        multi_class='multinomial',
                                        solver='saga',
                                        max_iter=10000,
                                        fit_intercept=False,
                                        tol=0.0005,
                                        random_state=np.random.randint(0, 999999))
        curr_model.fit(X[included], y.values.ravel())

        res = pd.DataFrame(
            {"# of var": len(included),
             "model": curr_model,
             "var_included": [included.copy()],
             "coefs": [curr_model.coef_.copy()],
             "Accuracy": best_score,
             "K-Fold Accuracy": kf_accuracy})
        total_res = pd.concat([total_res, res])
        if kf_accuracy > curr_best_kfold_accuracy:
            curr_best_kfold_accuracy = kf_accuracy
            changed = True
        else:
            j += 1
        if changed and verbose:
            print(f"Add {best_feature} with K-Fold Accuracy {kf_accuracy}")
        if j == 10:
            changed = False
        if not changed:
            break
    total_res = total_res.reset_index()
    return total_res                
                            
if __name__ == "__main__":
    total_time = time.time()

    spark = (SparkSession
            .builder
            .appName("ClusteringAnalysis")
            .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse")
            .config("hive.metastore.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse")
            .config("hive.exec.dynamic.partition.mode", "nonstrict")
            .config("hive.exec.dynamic.partition", "true")
            .enableHiveSupport()
            .getOrCreate())

    job_id = ""
    yparam = ""

    for i in range(1, len(sys.argv)):
        name = sys.argv[i].split(":")[0]
        value = sys.argv[i].split(":")[1]
        if name == "jobid": 
            job_id = value
        elif name == "area":
            cluster_area = value
        elif name == "yparam": 
            yparam = value

    if cluster_area == "local":
        hive_server = "localhost"
        presto_discovery = "localhost"
        hdfs_nn = "http://localhost:9870"
    elif cluster_area == "ich":
        hive_server = "ichbig-01-002"
        presto_discovery = "10.38.12.216"
        hdfs_nn = "http://icbig-00-11:9870"
    elif cluster_area == "wxh":
        hive_server = "wuxbigm-001-02"
        presto_discovery = "wuxbigm-004-01"
        hdfs_nn = "http://wuxbigm-001-01:50070"

    if cluster_area == "local":
        # 로컬 파일에서 직접 읽기
        local_path = f"/root/{job_id}"
        if os.path.exists(local_path):
            Total = pd.read_csv(local_path)
        else:
            # HDFS에서 읽기 (fallback)
            hdfsClient = InsecureClient(hdfs_nn)
            with hdfsClient.read('/user/hive/warehouse/input/'+job_id, encoding='utf-8') as reader:
                Total = pd.read_csv(reader)
    else:
        # HDFS에서 읽기 (원격 클러스터)
        hdfsClient = InsecureClient(hdfs_nn)
        with hdfsClient.read('/icbig/bizanal/fdcanalysis/'+job_id, encoding='utf-8') as reader:
            Total = pd.read_csv(reader)

    if Total.empty:
        print("Empty")
        insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
        insertQuery += create_insert_value('-1', 'error', 'error', 'error', job_id, job_id[0:8])
        spark.sql(insertQuery)
        sys.exit(0)

    elif (len(Total) < 10):
        insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
        insertQuery += create_insert_value('-4', 'error', 'error', 'error', job_id, job_id[0:8])
        spark.sql(insertQuery)
        sys.exit(0)
    else:
        path = "/User/james/PycharmProjets/test1/result/"

        response = yparam
        not_used = ['END_TIME', 'GROUP_ID']
        ETIME = ['END_TIME']

        if Total[response].std() == 0:
            insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
            insertQuery += create_insert_value('-2', 'error', 'error', 'error', job_id, job_id[0:8])
            spark.sql(insertQuery)
            sys.exit(0)

        Y2SM_si = Total.dropna(subset=[response])
        Y2SM_si.fillna(Y2SM_si.mean(numeric_only=True), inplace=True)

        x_features = [x_feature for x_feature in list(Y2SM_si.columns) if x_feature not in response]

        describes = Y2SM_si[x_features].describe()

        no_volatility_features = [k for k, v in describes.loc['std'].to_dict().items() if v == 0.0]

        unixtime_features =  filter_timestamp_columns(describes)

        x_features = [x_feature for x_feature in x_features if x_feature not in no_volatility_features + unixtime_features]

        temp_not_used_columns = list(set(not_used).union(set(ETIME)))
        x_features = [x_feature for x_feature in x_features if x_feature not in temp_not_used_columns]

        Y2SM1 = Y2SM_si.copy()
        X_Y2SM_si = Y2SM1[x_features]
        X_Y2SM_si = X_Y2SM_si.dropna(axis=1)
        y_Y2SM_si = Y2SM1[response]

        if X_Y2SM_si.shape[1] < 2:
            insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
            insertQuery += create_insert_value('-3', 'error', 'error', 'error', job_id, job_id[0:8])
            spark.sql(insertQuery)
            sys.exit(0)

        scaler = preprocessing.StandardScaler()
        X_Y2SM_si = pd.DataFrame(scaler.fit_transform(X_Y2SM_si), columns=X_Y2SM_si.columns)

        if pd.api.types.is_string_dtype(y_Y2SM_si.dtypes):
            y_Y2SM_si = y_Y2SM_si
        else:
            y_Y2SM_si = pd.DataFrame(scaler.fit_transform(y_Y2SM_si.values.reshape(-1, 1)), columns=[response])

        if len(np.unique(Total.GROUP_ID)) == 1:
            cluster_df = get_clustering_df(Y2SM_si, [response, ETIME[0]], 1)
            scaled_cluster_df = scaler.fit_transform(cluster_df[[response, 'Time1']])
            start_time = datetime.datetime.now()

            # 동적 파라미터 계산
            dynamic_params = get_dynamic_clustering_params(
                n_samples=len(X_Y2SM_si),
                n_features=len(X_Y2SM_si.columns)
            )
            print(f"[INFO] Dynamic Parameters: {dynamic_params}")

            # PCA로 변경 (성능 최적화)
            # n_components는 데이터의 feature 수를 초과할 수 없음
            n_components = min(dynamic_params['n_components'], scaled_cluster_df.shape[1])
            pca = PCA(n_components=n_components)
            y_pca_df = pd.DataFrame(pca.fit_transform(scaled_cluster_df))
            explained_variance = sum(pca.explained_variance_ratio_)
            print(f"[INFO] PCA: n_components={n_components}, explained_variance={explained_variance:.4f}")

            cluster_result = get_optimal_cluster_number(data=y_pca_df,
                                                        iterations=dynamic_params['iterations'],
                                                        max_cluster=dynamic_params['max_cluster'],
                                                        verbose=False)
            l = datetime.datetime.now()

            cluster_med = int(np.around(statistics.median(cluster_result)))
            print(f"[INFO] Optimal Cluster Number: {cluster_med}")

            # MiniBatchKMeans로 변경 (성능 최적화)
            km_FIN = MiniBatchKMeans(
                n_clusters=cluster_med,
                batch_size=min(100, len(y_pca_df)),
                random_state=42,
                n_init=20,
                max_iter=300,
                init='k-means++'
            )
            km_FIN1 = km_FIN.fit(y_pca_df)
            y_pca_df['Cluster'] = km_FIN1.labels_
            Target = km_FIN1.labels_
        else:
            le = preprocessing.LabelEncoder()
            le.fit(Y2SM_si[response].values)
            Target = le.transform(Y2SM_si[response].values)
            Y2SM_si.loc[:, response] = Target
            cluster_df = get_clustering_df(Y2SM_si, [response, ETIME[0]], 1)
            cluster_med = len(le.classes_)
            l = datetime.datetime.now()
        timestr2 = time.strftime("%Y%m%d-%H%M%S")

        # 동적 파라미터 로깅
        print(f"[INFO] Data size: {len(X_Y2SM_si)} samples × {len(X_Y2SM_si.columns)} features")


        result_rfc = pd.DataFrame()
        X_Y2SM_si = pd.DataFrame(X_Y2SM_si)
        X_Y2SM_si = X_Y2SM_si.dropna(axis=1)
        Target1 = pd.Series(Target, dtype='category')
        p = datetime.datetime.now()

        # 동적 파라미터: RF Bootstrap 횟수
        rf_iterations = dynamic_params['rf_iterations']
        print(f"[INFO] RF Bootstrap: {rf_iterations} iterations")

        for k in np.arange(rf_iterations):
            X_train, X_test, y_train, y_test = train_test_split(X_Y2SM_si, Target,
                                                                test_size=0.3,
                                                                random_state=np.random.randint(0, 999999))
            a = datetime.datetime.now()
            rf_classifier = RandomForestClassifier(n_estimators=100,
                                                   oob_score=True,
                                                   random_state=np.random.randint(0, 999999),
                                                   n_jobs = -1,
                                                   warm_start=True).fit(X_train, y_train)
            b = datetime.datetime.now()

            feature_impt = rf_classifier.feature_importances_
            feature_impt_std = np.std([tree.feature_importances_ for tree in rf_classifier.estimators_], axis=0)

            rfc_result = pd.DataFrame().assign(**{'variable': X_train.columns,
                                                  'importance': feature_impt,
                                                  'std': feature_impt_std})
            
            scaled_importance = rfc_result['importance'].astype(np.float64) / rfc_result['std'].astype(np.float64)
            scaled_importance = scaled_importance.astype(np.float64)

            rfc_result = rfc_result.assign(**{'s': scaled_importance,
                                              's_rank': scaled_importance.rank(),
                                              'i_rank': rfc_result.importance.rank(),
                                              'iteration': k})
                                              
            result_rfc = pd.concat((result_rfc, rfc_result), axis=0)
                                              
        q = datetime.datetime.now()
        result_rfc.importance = result_rfc["importance"].astype('float64')
        tests_RFC = result_rfc.groupby('variable')['importance'].describe()
        tests_RFC["standard_importance_rank"] = tests_RFC['mean'].rank(ascending=False)
        tests_RFC = tests_RFC.sort_values(by='standard_importance_rank', ascending=True)

        timestr0 = time.strftime("%Y%m%d-%H%M%S")
        FIN_Raw = pd.concat([X_Y2SM_si, cluster_df], axis=1)

        tests_RFC_mean_over_zeros = tests_RFC.query('mean > 0')

        # RF Pre-screening: 상위 N개 feature만 선택
        rf_top_n = dynamic_params['rf_top_n']
        if len(tests_RFC_mean_over_zeros) > rf_top_n:
            rf_prescreened = tests_RFC_mean_over_zeros.nlargest(rf_top_n, 'mean')
            print(f"[INFO] RF Pre-screening: {len(tests_RFC_mean_over_zeros)} → {rf_top_n} features")
        else:
            rf_prescreened = tests_RFC_mean_over_zeros
            print(f"[INFO] RF All {len(tests_RFC_mean_over_zeros)} features retained (less than RF_TOP_N)")


        feature = rf_prescreened.index.values
        f1 = list(feature)
        XX = X_Y2SM_si[f1]
        XX = XX.loc[:, XX.std() > .0]
        XX = pd.DataFrame(preprocessing.normalize(XX, axis=0), columns=XX.columns)
        h = datetime.datetime.now()

        try:
            if len(XX) < 30:
                logit_model = LogisticRegression(fit_intercept=False,
                                                 penalty = 'l2',
                                                 solver = 'saga',
                                                 multi_class = 'multinomial',
                                                 max_iter=10000,
                                                 tol=0.0005, random_state=np.random.randint(0, 999999))
            else:
                if len(np.unique(Target)) > 2:
                    logit_model = LogisticRegression(fit_intercept=False,
                                                     penalty='l2',
                                                     solver='saga',
                                                     multi_class='multinomial',
                                                     max_iter=10000,
                                                     tol=0.0005, random_state=np.random.randint(0, 999999))
                else:
                    logit_model = LogisticRegression(fit_intercept=False,
                                                     penalty='l2',
                                                     solver='saga',
                                                     multi_class='auto',
                                                     max_iter=10000,
                                                     tol=0.0005, random_state=np.random.randint(0, 999999))
            logit_model.fit(XX, Target.ravel())
        except:
            logit_model = LogisticRegression(fit_intercept=False,
                                             penalty='l2',
                                             solver='saga',
                                             multi_class='multinomial',
                                             max_iter=10000,
                                             tol=0.0005, random_state=np.random.randint(0, 999999))
            logit_model.fit(XX, Target.ravel())

        G = datetime.datetime.now()
        train_score = logit_model.score(XX, Target.ravel())
        logit_coefficients = logit_model.coef_.tolist()
        logit_coefficients = np.sign(logit_coefficients) * np.exp(logit_coefficients)

        aList = []
        for i in range(0, len(logit_coefficients)):
            for y in range(0, len(logit_coefficients[i])):
                if logit_coefficients[i][y] != 0 and logit_coefficients[i][y] != -0 and not (y in aList):
                    aList.append(y)

        aList.sort()
        importanceList = []
        for i in range(0, len(logit_coefficients)):
            tmp = []
            for y in aList:
                tmp.append(logit_coefficients[i][y])
            importanceList.append(tmp)

        importanceList = pd.DataFrame(importanceList)
        importanceList.columns = XX.columns[aList]
        RF_MNLCV_features_1st = pd.DataFrame(np.concatenate((pd.DataFrame(XX.columns[aList]),
                                                            importanceList.T), axis=1))
        RF_MNLCV_features_1st.index = XX.columns[aList]
        RF_MNLCV_features_1st = RF_MNLCV_features_1st.rename(columns={0: "Variable"})
        if len(np.unique(Total.GROUP_ID)) == 1:
            avg_score_abs_coeffs = np.sum(
                np.abs(RF_MNLCV_features_1st[[i + 1 for i in range(cluster_med)]]), axis=1) / cluster_med
        else:
            avg_score_abs_coeffs = RF_MNLCV_features_1st[1]
        RF_MNLCV_features_1st = RF_MNLCV_features_1st.assign(**({
            'Average Score of absolute coefficients': avg_score_abs_coeffs}))
        RF_MNLCV_features_1st = RF_MNLCV_features_1st.sort_values(by="Average Score of absolute coefficients",
                                                                  ascending=False)

        Target = pd.DataFrame(Target)
        if len(RF_MNLCV_features_1st) > 0:
            feature = RF_MNLCV_features_1st['Variable']
        else:
            feature = tests_RFC_mean_over_zeros.index

        f1 = list(feature)

        # Post-Logistic Pre-screening: 상위 N개 feature만 선택 (대용량 데이터에서만 적용)
        post_logistic_top_n = dynamic_params['post_logistic_top_n']
        if post_logistic_top_n is not None and len(f1) > post_logistic_top_n:
            f1 = f1[:post_logistic_top_n]
            print(f"[INFO] Post-Logistic Pre-screening: {len(feature)} → {post_logistic_top_n} features")
        else:
            if post_logistic_top_n is None:
                print(f"[INFO] Post-Logistic Pre-screening skipped (small dataset: {len(X_Y2SM_si)} samples)")
            else:
                print(f"[INFO] All {len(f1)} Logistic features retained (less than POST_LOGISTIC_TOP_N)")

        XX = X_Y2SM_si[f1]
        XX = XX.loc[:, XX.std() > .0]
        XX = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(XX), columns=XX.columns)
        I = datetime.datetime.now()

        # 동적 파라미터: K-Fold 설정
        kfold_params = get_dynamic_kfold_params(len(XX))
        print(f"[INFO] Step Forward K-Fold: k={kfold_params['k']}, tol={kfold_params['tol']}")
        RF_MNLCV_KFold = step_forward_k_fold(XX, Target, k=kfold_params['k'],
                                             accuracy_update_tol=kfold_params['tol'], verbose=True)
        J = datetime.datetime.now()

        max_index = RF_MNLCV_KFold['K-Fold Accuracy'].idxmax()
        RF_MNLCV_KFold_features = RF_MNLCV_KFold['var_included'][max_index]
        RF_MNLCV_KFold_coef = RF_MNLCV_KFold['coefs'][max_index].T
        RF_MNLCV_KFold_2nd = pd.DataFrame(np.column_stack((RF_MNLCV_KFold_features, RF_MNLCV_KFold_coef)))      
        RF_MNLCV_KFold_2nd.iloc[:, 1:] = RF_MNLCV_KFold_2nd.iloc[:, 1:].astype(float) 
        RF_MNLCV_KFold_2nd = RF_MNLCV_KFold_2nd.rename(columns={0: "Variable"})

        if len(np.unique(Total.GROUP_ID)) == 1:
            avg_score_abs_coeffs = np.sum(
                np.abs(RF_MNLCV_KFold_2nd[[i + 1 for i in range(cluster_med)]]), axis=1) / cluster_med
        else:
            avg_score_abs_coeffs = RF_MNLCV_KFold_2nd[1]
        RF_MNLCV_KFold_2nd = RF_MNLCV_KFold_2nd.assign(**({
            'Average Score of absolute coefficients': avg_score_abs_coeffs}))
        RF_MNLCV_KFold_2nd = RF_MNLCV_KFold_2nd.sort_values(by="Average Score of absolute coefficients",
                                                            ascending=False)

        K = datetime.datetime.now()

        def compute_spearman_correlations(feature_pair):
            final_feature_name, suggestion_feature_name = feature_pair
            corr_value, _ = stats.spearmanr(FIN_Raw[suggestion_feature_name],
                                            FIN_Raw[final_feature_name])
            res_corr_value, _ = stats.spearmanr(FIN_Raw[[response]],
                                                FIN_Raw[suggestion_feature_name])
            return (final_feature_name, suggestion_feature_name,
                    abs(corr_value), abs(res_corr_value))

        def compute_mine_correlation_parallel(feature):
            mine = MINE()
            mine.compute_score(FIN_Raw[response], FIN_Raw[feature])
            mic_value = mine.mic()
            return (feature, mic_value)

        feature_pairs = [(final_feature_name, suggestion_feature_name)
                         for final_feature_name in RF_MNLCV_KFold_2nd.Variable.values
                         for suggestion_feature_name in tests_RFC_mean_over_zeros.index.values.ravel()]

        # Sequential processing (faster for small datasets)
        spearman_result = [compute_spearman_correlations(pair) for pair in feature_pairs]

        RF_MNLCV_KFold_1st_FIN_corr = pd.DataFrame(spearman_result,
                                                     columns=['Final', 'Suggestion',
                                                              'Absolute Spearman Corr.(Final-Suggestion)',
                                                              'Absolute Spearman Corr.(Target-Suggestion)'])

        RF_MNLCV_KFold_1st_FIN_corr = RF_MNLCV_KFold_1st_FIN_corr.loc[
                                      RF_MNLCV_KFold_1st_FIN_corr['Absolute Spearman Corr.(Target-Suggestion)'] >
                                      RF_MNLCV_KFold_1st_FIN_corr['Absolute Spearman Corr.(Target-Suggestion)'].median(), :]

        unique_features = RF_MNLCV_KFold_1st_FIN_corr.iloc[:, 1].unique()
        # Sequential processing (faster for small datasets)
        mine_results = [compute_mine_correlation_parallel(feat) for feat in unique_features]
        mine_results = dict(mine_results)

        mic_total_res = [mine_results[RF_MNLCV_KFold_1st_FIN_corr.iloc[i, 1]]
                        for i in range(RF_MNLCV_KFold_1st_FIN_corr.shape[0])]

        RF_MNLCV_KFold_1st_FIN_corr['Maximal Information Coeff.(Target-Suggestion)'] = mic_total_res
        RF_MNLCV_KFold_1st_FIN_corr['mean_of_corr'] = (RF_MNLCV_KFold_1st_FIN_corr[
                                                            'Absolute Spearman Corr.(Final-Suggestion)'] +
                                                          RF_MNLCV_KFold_1st_FIN_corr[
                                                            'Absolute Spearman Corr.(Target-Suggestion)']) / 2
        RF_MNLCV_KFold_1st_FIN_corr['mean_of_corr(mic)'] = (RF_MNLCV_KFold_1st_FIN_corr[
                                                                   'Absolute Spearman Corr.(Final-Suggestion)'] +
                                                               RF_MNLCV_KFold_1st_FIN_corr[
                                                                   'Maximal Information Coeff.(Target-Suggestion)']) / 2
        RF_MNLCV_KFold_1st_FIN_corr = RF_MNLCV_KFold_1st_FIN_corr.sort_values(by='mean_of_corr(mic)', ascending=False)
        RF_MNLCV_KFold_1st_FIN_corr = RF_MNLCV_KFold_1st_FIN_corr.reset_index().drop(columns=['index'])
        
        index_list = []
        for i in RF_MNLCV_KFold_1st_FIN_corr.Suggestion.unique():
            index_list.append(
                RF_MNLCV_KFold_1st_FIN_corr.loc[
                    RF_MNLCV_KFold_1st_FIN_corr.Suggestion == i, 'mean_of_corr(mic)'].idxmax())
        RF_MNLCV_KFold_1st_FIN_corr = RF_MNLCV_KFold_1st_FIN_corr.iloc[index_list, :].reset_index()
        m = datetime.datetime.now()
                
        result_code = "0" if RF_MNLCV_KFold_1st_FIN_corr.size > 0 else "2"

        insertQuery ="Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
        for idx, row in tests_RFC.iterrows():
            step = idx.split(".")[0]
            param = idx.split(".")[1]
            insertQuery += create_insert_value(result_code, param, step, row['mean'], job_id, job_id[0:8]) + ","
        insertQuery = insertQuery[0:len(insertQuery) - 1]
        spark.sql(insertQuery)

        insertQuery = "Insert into bizanal.fdcalysis_coeff VALUES"
        for idx, row in RF_MNLCV_KFold_2nd.iterrows():
            insertQuery += create_insert_value(result_code,
                                               row['Variable'],
                                               result_code,
                                               row['Average Score of absolute coefficients'],
                                               job_id, job_id[0:8]) + ","
        insertQuery = insertQuery[0:len(insertQuery)-1]
        spark.sql(insertQuery)

        if result_code == "0":
            insertQuery = "Insert into bizanal.fdcanalysis_final VALUES"
            for idx, row in RF_MNLCV_KFold_1st_FIN_corr.iterrows():
                insertQuery += create_insert_value('0',
                                                row['Final'], row['Suggestion'],
                                                row['Absolute Spearman Corr.(Final-Suggestion)'],
                                                row['Absolute Spearman Corr.(Target-Suggestion)'],
                                                row['Maximal Information Coeff.(Target-Suggestion)'],
                                                row['mean_of_corr'],
                                                row['mean_of_corr(mic)'],
                                                job_id, job_id[0:8]) + ","
            insertQuery = insertQuery[0:len(insertQuery)-1]
            spark.sql(insertQuery)
            




                    
                    
                    
                    
                    

                                           
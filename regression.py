import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import statsmodels.regression.linear_model as sm
import warnings

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from minepy import MINE
from dateutil.relativedelta import *
from joblib import Parallel, delayed

from pyhive import presto
from pyhive import hive
from pyspark.sql import SparkSession
from hdfs import InsecureClient
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Dask for distributed Random Forest
from dask.distributed import Client
from joblib import parallel_backend

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

ERROR_CODE_01 = '-1'
ERROR_CODE_02 = '-2'
ERROR_CODE_03 = '-3'
ERROR_CODE_04 = '1'
ERROR_CODE_05 = '-4'
SUCCESS_CODE_01 = '0'
SUCCESS_CODE_02 = '2'

def create_insert_value(*str_values):
    str_query = " ("
    for idx, value in enumerate(str_values):
        if idx == len(str_values)-1:
            str_query += "'" + str(value) + "')"
        else:
            str_query += "'" + str(value) + "',"
    return str_query

def cal_adjusted_r_squared(r_squared, n, p):
    value = 1 - (((1 - r_squared) * (n - 1) / (n - p - 1)))
    return value

def bootstrap_rf(X, Y, test_size=0.3, n_est=100):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                        test_size=test_size,
                                                        random_state=np.random.randint(0, 999999))
    start_time = (datetime.datetime.now())
    rf_regressor = RandomForestRegressor(n_estimators=n_est,
                                         oob_score=True,
                                         random_state=np.random.randint(0, 999999),
                                         n_jobs=-1,
                                         warm_start=True).fit(X_train, y_train.values.ravel())
    end_time = (datetime.datetime.now())

    feature_impt = rf_regressor.feature_importances_
    feature_impt_std = np.std([tree.feature_importances_ for tree in rf_regressor.estimators_], axis=0)

    rfc_result = pd.DataFrame().assign(**{'variable': X_train.columns,
                                          'importance': feature_impt,
                                          'std': feature_impt_std})

    scaled_importance = rfc_result['importance'].astype(np.float64) / rfc_result['std'].astype(np.float64)
    scaled_importance = scaled_importance.astype(np.float64)
 
    rfc_result = rfc_result.assign(**{'s': scaled_importance,
                                            's_rank': scaled_importance.rank(),
                                            'i_rank': rfc_result.importance.rank()})
    return rfc_result

def lasso_scores(alpha, train_x, train_y, test_x, test_y):
    lasso = Lasso(alpha=alpha,
                  tol=0.0005,
                  max_iter=1000,
                  random_state=42).fit(train_x, train_y)
    train_score = lasso.score(train_x, train_y)
    test_score = lasso.score(test_x, test_y)
    return alpha, train_score, test_score

def create_lasso_dataset(df, features):
    data = df[features]
    data = data.loc[:, data.std() > .0]
    data = pd.DataFrame(preprocessing.normalize(data, axis=0), columns=data.columns)
    return data

def evaluate_feature_addition(column, included, X, y):
    """병렬 평가용 헬퍼 함수"""
    train_features = included + [column]
    train_x = X[train_features]
    model = LinearRegression(fit_intercept=False, n_jobs=-1)
    model.fit(train_x, y)
    score = model.score(train_x, y)

    if (len(X) - len(model.coef_) - 1) == 0:
        adj_score = score
    else:
        adj_score = cal_adjusted_r_squared(score, len(X), len(model.coef_))
    return column, adj_score

def step_forward_k_fold(X, y, k=5, tol=0.000, verbose=True):
    """병렬화된 Step Forward K-Fold Selection"""
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    included = list()
    total_result = pd.DataFrame(
        columns=["# of var", "model", "var_included", "coefs", "Adj_Rsquared", "K-Fold Adj_Rsquared"])

    curr_best_score = -np.inf
    curr_best_kfold_score = -np.inf
    j = 0

    while True:
        changed = False
        included.sort()
        excluded = [feature for feature in list(X.columns) if feature not in included]
        excluded.sort()

        if len(excluded) == 0:
            print("Stopping the Forward Selection")
            break

        # === 병렬 평가: 모든 excluded feature를 동시에 평가 ===
        if verbose:
            print(f"[Step {len(included)+1}] Evaluating {len(excluded)} features in parallel...")
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_feature_addition)(col, included, X, y) for col in excluded
        )
        new_score_dict = {col: score for col, score in results}
        best_feature, best_adj_score = max(results, key=lambda x: x[1])

        if verbose:
            print(f"  Scores: {new_score_dict}")
            print(f"  Best feature: {best_feature} (Adj-R²={best_adj_score:.6f})")

        if (best_adj_score - curr_best_score) > tol:
            included.append(best_feature)
            curr_best_score = best_adj_score
            changed = True

        cv_result = pd.DataFrame(columns=['k_fold', 'Adj_Rsquared'])
        i = 0

        if len(X) >= 30:
            for _, test_idx in kf.split(X, y):
                X_test = X.iloc[test_idx,]
                y_test = y.iloc[test_idx,]
                model = LinearRegression(fit_intercept=False,
                                         n_jobs=-1)
                model.fit(X_test[included + [best_feature]], y_test)
                score = model.score(X_test[included + [best_feature]], y_test)
                cv_score = cal_adjusted_r_squared(score, len(X_test), len(model.coef_))
                cv_inner_res = pd.DataFrame({'k_fold': [i],
                                              'Adj_Rsquared': [cv_score]})
                cv_result = pd.concat([cv_result, cv_inner_res])
                i += 1
        else:
            model = LinearRegression(fit_intercept=False,
                                     n_jobs=-1)
            model.fit(X[included + [best_feature]], y)
            score = model.score(X[included + [best_feature]], y)
            if (len(X) - len(model.coef_) - 1) == 0:
                cv_score = score
            else:
                cv_score = cal_adjusted_r_squared(score, len(X), len(model.coef_))
            cv_inner_res = pd.DataFrame({"k_fold": [0], "Adj_Rsquared": [cv_score]})
            cv_result = pd.concat([cv_result, cv_inner_res])

        kf_score = cv_result['Adj_Rsquared'].mean()
        curr_model = LinearRegression(fit_intercept=False,
                                      n_jobs=-1).fit(X[included], y)                            

        if kf_score > curr_best_kfold_score:
            curr_best_kfold_score = kf_score
        else:
            j += 1
        if changed and verbose:
            pass
        if j == 10:
            changed = False
        if not changed:
            break

        result = pd.DataFrame(
            {'# of var': len(included),
              'model': curr_model,
              'var_included': [included.copy()],
              'coefs': [curr_model.coef_.copy()],
              'Adj_Rsquared': best_adj_score,
              'K-Fold Adj_Rsquared': kf_score})
        total_result = pd.concat([total_result, result])
    total_result.reset_index(inplace=True)
    return total_result

if __name__ == "__main__":
    total_time = time.time()

    spark = (SparkSession
            .builder
            .appName("RegressionAnalysis")
            .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse")
            .config("hive.metastore.warehouse.dir", "hdfs://localhost:9000/user/hive/warehouse")
            .config("hive.exec.dynamic.partition.mode", "nonstrict")
            .enableHiveSupport()
            .getOrCreate())

    job_id = ""
    yparam = ""
    cluster_area = ""

    for i in range(1, len(sys.argv)):
        name = sys.argv[i].split(":")[0]
        value = sys.argv[i].split(":")[1]
        if (name == "jobid"): 
            job_id = value
        elif (name == "area"):
            cluster_area = value
        elif (name == "yparam"): 
            yparam = value

    if (cluster_area == "local"):
        hive_server = "localhost"
        presto_discovery = "localhost"
        hdfs_nn = "http://localhost:9870"
    elif (cluster_area == "ich"):
        hive_server = "ichbig-01-002"
        presto_discovery = "10.38.12.216"
        hdfs_nn = "http://icbig-00-11:9870"
    elif (cluster_area == "wxh"):
        hive_server = "wuxbigm-001-02"
        presto_discovery = "wuxbigm-004-01"
        hdfs_nn = "http://wuxbigm-001-01:50070"

    # Local file support for cluster_area == "local"
    if cluster_area == "local":
        local_path = f"/root/{job_id}"
        if os.path.exists(local_path):
            print(f"[INFO] Reading from local file: {local_path}")
            Total = pd.read_csv(local_path)
        else:
            print(f"[INFO] Local file not found, reading from HDFS")
            hdfsClient = InsecureClient(hdfs_nn)
            with hdfsClient.read('/user/hive/warehouse/input/'+job_id, encoding='utf-8') as reader:
                Total = pd.read_csv(reader)
    else:
        hdfsClient = InsecureClient(hdfs_nn)
        with hdfsClient.read('/user/hive/warehouse/input/'+job_id, encoding='utf-8') as reader:
            Total = pd.read_csv(reader)

    if Total.empty:
        print("Empty")
        insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
        insertQuery += create_insert_value(ERROR_CODE_01, 'error', 'error', 'error', job_id, job_id[0:8])
        spark.sql(insertQuery)
        sys.exit(0)

    elif (len(Total) < 10):
        insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
        insertQuery += create_insert_value(ERROR_CODE_05, 'error', 'error', 'error', job_id, job_id[0:8])
        spark.sql(insertQuery)
        sys.exit(0)
    else:
        path = "/User/james/PycharmProjets/test1/result/"

        response = yparam
        not_used = {'END_TIME', 'END_TM', 'FAB', 'LOT_CD', 'ALIAS_LOT_ID', 'LOT_ID', 'WF_ID', 'group_key', 'module', 'recipe_id', 'oper', 'FAB(FDC)', 'LOT_CD(FDC)'}
        list_end_time = ['END_TIME']
        str_end_time = "END_TIME"

        if Total[response].std() == 0:
            insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
            insertQuery += create_insert_value(ERROR_CODE_02, 'error', 'error', 'error', job_id, job_id[0:8])
            spark.sql(insertQuery)
            sys.exit(0)

        R2SM_si = Total.dropna(subset=[response])
        R2SM_si.fillna(R2SM_si.mean(numeric_only=True), inplace=True)

        x_features = [x_feature for x_feature in list(R2SM_si.columns) if x_feature not in response]
        describes = R2SM_si[x_features].describe()
        no_volatility_features = [k for k, v in describes.loc['std'].to_dict().items() if v == 0.0 or np.isnan(v)]
        x_features = [x_feature for x_feature in x_features if x_feature not in no_volatility_features]

        temp_not_used_columns = not_used.union(list_end_time)
        x_features = [x_feature for x_feature in x_features if x_feature not in temp_not_used_columns]

        X_R2SM_si = R2SM_si[x_features]
        X_R2SM_si = X_R2SM_si.dropna(axis=1)
        y_R2SM_si = R2SM_si[response]

        if X_R2SM_si.empty :
            insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
            insertQuery += create_insert_value(ERROR_CODE_03, 'error', 'error', 'error', job_id, job_id[0:8])
            spark.sql(insertQuery)
            sys.exit(0)

        scaler = preprocessing.StandardScaler()
        X_R2SM_si = pd.DataFrame(scaler.fit_transform(X_R2SM_si), columns=X_R2SM_si.columns)
        y_R2SM_si = pd.DataFrame(scaler.fit_transform(y_R2SM_si.values.reshape(-1,1)), columns=[response])

        print(f"[INFO] Data loaded: {len(X_R2SM_si)} rows × {len(X_R2SM_si.columns)} features")

        if np.all(X_R2SM_si.std() == 0):
            insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
            insertQuery += create_insert_value(ERROR_CODE_03, 'error', 'error', 'error', job_id, job_id[0:8])
            spark.sql(insertQuery)
            sys.exit(0)

        print(f"[INFO] Step 1: Random Forest Bootstrap (10 iterations) with Dask...")

        # === Pre-screening 파라미터 설정 (README 1829-1951행 참고) ===
        # 데이터가 1만 건이면 feature 수가 많을 수 있으므로 상위 N개로 제한
        RF_TOP_N = 100  # README에서 제안된 대규모 데이터용 설정
        LASSO_TOP_N = 50  # Lasso에서 선택할 최대 feature 수

        # === Dask 분산 처리 적용 ===
        client = Client(n_workers=2, threads_per_worker=2, memory_limit='2GB', silence_logs=True)
        try:
            with parallel_backend('dask'):
                out = Parallel(n_jobs=-1)(delayed(bootstrap_rf)(X_R2SM_si, y_R2SM_si, 0.3) for _ in range(10))
        finally:
            client.close()

        result_rfc = pd.concat(out, axis=0, ignore_index=True)

        stats_of_importance_each_variable = result_rfc.groupby('variable')['importance'].describe()
        stats_of_importance_each_variable['standard_importance_rank'] = stats_of_importance_each_variable['mean'].rank(ascending=False)
        stats_of_importance_each_variable.sort_values(by='standard_importance_rank',
                                                      ascending=True,
                                                      inplace=True)
        response_df = R2SM_si[[response, str_end_time]].reset_index()

        response_df.loc[:, 'TIME'] = pd.to_datetime(response_df.loc[:, str_end_time])
        response_df.loc[:, 'TIME1'] = pd.to_numeric(response_df.loc[:, 'TIME'])

        final_raw_df = pd.concat([X_R2SM_si, response_df], axis=1)
        rf_feature_impt_mean_over_zero = stats_of_importance_each_variable.query('mean > 0')
        print(f"[INFO] Random Forest completed: {len(rf_feature_impt_mean_over_zero)} features selected")

        # === Pre-screening: 상위 N개 feature만 선택 ===
        if len(rf_feature_impt_mean_over_zero) > RF_TOP_N:
            rf_feature_impt_mean_over_zero = rf_feature_impt_mean_over_zero.nlargest(RF_TOP_N, 'mean')
            print(f"[INFO] Pre-screening: Selected top {RF_TOP_N} features from Random Forest")
        else:
            print(f"[INFO] All {len(rf_feature_impt_mean_over_zero)} RF features retained (less than RF_TOP_N)")

        features = list(rf_feature_impt_mean_over_zero.index.values)
        XX_R2SM_si = create_lasso_dataset(X_R2SM_si, features)
        candidate_alphas = np.logspace(-10, -3, 100)  # 0.0000000001 ~ 0.001

        print(f"[INFO] Step 2: LASSO Regression started with {len(XX_R2SM_si.columns)} features...")

        X_train, X_test, y_train, y_test = train_test_split(XX_R2SM_si, y_R2SM_si, test_size=0.3, random_state=42)
        lasso_performance = \
            Parallel(n_jobs=-1, prefer='threads')(
                delayed(lasso_scores)(alpha, X_train, y_train, X_test, y_test) for alpha in candidate_alphas)
        lasso_performance = pd.DataFrame(lasso_performance, columns=['alpha', 'train_score', 'test_score'])
        optimal_alpha = lasso_performance.iloc[lasso_performance.iloc[:, 2].idxmax(), 0]

        print("Optimal alpha for LASSO: {: .6f}".format(optimal_alpha))

        optimal_lasso_model = Lasso(fit_intercept=False,
                                    tol=0.0005,
                                    alpha=optimal_alpha).fit(XX_R2SM_si, y_R2SM_si)
                
        score = optimal_lasso_model.score(XX_R2SM_si, y_R2SM_si)
        adj_score = cal_adjusted_r_squared(score,
                                        len(XX_R2SM_si),
                                        len(optimal_lasso_model.coef_[abs(optimal_lasso_model.coef_) > .0]))

        lasso_coefficients = optimal_lasso_model.coef_.tolist()
        selected_coefficients_idx = np.where(np.abs(lasso_coefficients) != 0)[0].tolist()
        selected_coefficients_values = np.take(lasso_coefficients, selected_coefficients_idx)

        # === Pre-screening: 상위 N개 coefficient만 선택 ===
        if len(selected_coefficients_idx) > LASSO_TOP_N:
            # 절대값이 큰 순서대로 상위 N개 선택
            top_idx = np.argsort(np.abs(selected_coefficients_values))[-LASSO_TOP_N:]
            selected_coefficients_idx = np.array(selected_coefficients_idx)[top_idx].tolist()
            selected_coefficients_values = np.take(lasso_coefficients, selected_coefficients_idx)
            print(f"[INFO] Pre-screening: Selected top {LASSO_TOP_N} Lasso features")
        else:
            print(f"[INFO] All {len(selected_coefficients_idx)} Lasso features retained")

        print(f"[INFO] LASSO completed: {len(selected_coefficients_idx)} features with non-zero coefficients")

        if len(selected_coefficients_idx) == 0:
            insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
            for idx, row in stats_of_importance_each_variable.iterrows():
                step = idx.split(".")[0]
                param = idx.split(".")[1]
                insertQuery += create_insert_value(ERROR_CODE_04, 
                                                   param, step, row['mean'], job_id, job_id[0:8]) + ","
            insertQuery = insertQuery[0:len(insertQuery)-1]
            spark.sql(insertQuery)
        else:
            lasso_features_1st = pd.DataFrame().assign(**{
                'variable': XX_R2SM_si.columns[selected_coefficients_idx],
                'score': selected_coefficients_values,
                'abs_score': abs(selected_coefficients_values)
            })
            lasso_features_1st.sort_values(by='abs_score', ascending=False, inplace=True)

            str_time_0 = time.strftime("%Y%m%d-%H%M%S")

            features = list(lasso_features_1st['variable'])
            XX_R2SM_si = create_lasso_dataset(X_R2SM_si, features)

            start_time_5fold = datetime.datetime.now()
            print(f"[INFO] Step 3: Step Forward K-Fold Selection started with {len(XX_R2SM_si.columns)} features...")
            total_result = step_forward_k_fold(XX_R2SM_si, y_R2SM_si)
            end_time_5fold = datetime.datetime.now()
            elapsed_5fold = (end_time_5fold - start_time_5fold).total_seconds()

            max_adj_rsquared_idx = total_result['K-Fold Adj_Rsquared'].idxmax()

            lasso_forward = pd.DataFrame().assign(**{
                'Variable': total_result['var_included'][max_adj_rsquared_idx],
                'Score': total_result['coefs'][max_adj_rsquared_idx].ravel(),
                'ABS_Score': np.abs(total_result['coefs'][max_adj_rsquared_idx].ravel())})
            lasso_forward.sort_values(by='ABS_Score', ascending=False, inplace=True)
            print(f"[INFO] Step Forward completed in {elapsed_5fold:.2f}s: {len(lasso_forward['Variable'].tolist())} features selected")

            str_time_0 = time.strftime("%Y%m%d-%H%M%S")

            model = total_result['model'][max_adj_rsquared_idx]
            score = model.score(XX_R2SM_si[total_result['var_included'][max_adj_rsquared_idx]], y_R2SM_si)

            non_zero_coefficients = model.coef_[abs(model.coef_) > 0]
            train_adj_rsquared = cal_adjusted_r_squared(score, len(XX_R2SM_si), len(non_zero_coefficients))

            start_time_corr = datetime.datetime.now()

            lasso_forward_features = list(lasso_forward.Variable)
            rf_features = rf_feature_impt_mean_over_zero.index.values.ravel()


            def compute_spearman_correlations(feature_pair):
                lasso_feature_name, rf_feature_name = feature_pair
                corr_value, p_value = stats.spearmanr(final_raw_df[rf_feature_name],
                                                      final_raw_df[lasso_feature_name])
                res_corr_value, res_p_value = stats.spearmanr(final_raw_df[[response]],
                                                              final_raw_df[rf_feature_name])
                return (lasso_feature_name, rf_feature_name,
                        abs(corr_value), p_value, abs(res_corr_value), res_p_value)
            
            def compute_mine_correlation_parallel(feature):
                mine = MINE()
                mine.compute_score(final_raw_df[response], final_raw_df[feature])
                mic_value = mine.mic()
                return (feature, mic_value)

            feature_pairs = [(lasso_feature_name, rf_feature_name)
                             for lasso_feature_name in lasso_forward_features
                             for rf_feature_name in rf_features]

            print(f"[INFO] Step 4: MIC/Spearman Correlation started with {len(feature_pairs)} pairs (sequential processing)...")
            # Sequential processing (faster for small datasets)
            spearman_result = [compute_spearman_correlations(pair) for pair in feature_pairs]
            print(f"[INFO] Spearman correlation completed: {len(spearman_result)} pairs processed")

            rf_and_lasso_feature_corr = pd.DataFrame(spearman_result,
                                                     columns=['Final', 'Suggestion',
                                                              'Absolute Spearman Corr.(Final-Suggestion)',
                                                              'P-value of Absolute Spearman Corr.(Final-Suggestion)',
                                                              'Absolute Spearman Corr.(Target-Suggestion)',
                                                              'P-value of Absolute Spearman Corr.(Target-Suggestion)'])

            rf_and_lasso_feature_corr = rf_and_lasso_feature_corr.loc[
                                        rf_and_lasso_feature_corr['Absolute Spearman Corr.(Target-Suggestion)'] >
                                        rf_and_lasso_feature_corr['Absolute Spearman Corr.(Target-Suggestion)'].median(), :]


            unique_features = rf_and_lasso_feature_corr.iloc[:, 1].unique()
            print(f"[INFO] MIC calculation started for {len(unique_features)} unique features (sequential processing)...")
            # Sequential processing (faster for small datasets)
            mine_results = [compute_mine_correlation_parallel(feat) for feat in unique_features]
            mine_results = dict(mine_results)
            print(f"[INFO] MIC calculation completed: {len(mine_results)} features processed")

            mic_total_result = [mine_results[rf_and_lasso_feature_corr.iloc[i, 1]]
                                for i in range(rf_and_lasso_feature_corr.shape[0])]


            rf_and_lasso_feature_corr['Maximal Information Coeff.(Target-Suggestion)'] = mic_total_result
            mean_of_corr = (rf_and_lasso_feature_corr['Absolute Spearman Corr.(Final-Suggestion)'] +
                            rf_and_lasso_feature_corr['Absolute Spearman Corr.(Target-Suggestion)'])/2
            mean_of_corr_mic = (rf_and_lasso_feature_corr['Absolute Spearman Corr.(Final-Suggestion)'] +
                                rf_and_lasso_feature_corr['Maximal Information Coeff.(Target-Suggestion)'])/2

            rf_and_lasso_feature_corr['mean_of_corr'] = mean_of_corr
            rf_and_lasso_feature_corr['mean_of_corr(mic)'] = mean_of_corr_mic
            rf_and_lasso_feature_corr.sort_values(by='mean_of_corr(mic)', ascending=False, inplace=True)
            rf_and_lasso_feature_corr = rf_and_lasso_feature_corr.reset_index().drop(columns=['index'])

            mic_max_idx_each_feature = sorted(rf_and_lasso_feature_corr.groupby('Suggestion')['mean_of_corr(mic)'].idxmax().values)
            rf_and_lasso_feature_corr_final = rf_and_lasso_feature_corr.iloc[mic_max_idx_each_feature, :].reset_index()

            end_time_corr = datetime.datetime.now()

            result_code = SUCCESS_CODE_01 if rf_and_lasso_feature_corr_final.size > 0 else SUCCESS_CODE_02

            print(f"[INFO] Step 5: Saving results to Hive tables started...")
            print(f"[INFO] Final correlation results: {len(rf_and_lasso_feature_corr_final)} rows")

            labs = math.ceil(len(stats_of_importance_each_variable)/500.0)
            for i in range(labs):
                if (i + 1) * 500 < len(stats_of_importance_each_variable):
                    max_y = (i + 1) * 500
                else:
                    max_y = len(stats_of_importance_each_variable)
                insertQuery = "Insert into bizanal.fdcanalysis_rfimportance_v1 VALUES"
                for idx, row in stats_of_importance_each_variable[i*500:max_y].iterrows():
                    step = idx.split(".")[0]
                    param = idx.split(".")[1]
                    insertQuery += create_insert_value(result_code, param, step, row['mean'], job_id, job_id[0:8]) +","
            insertQuery = insertQuery[0:len(insertQuery)-1]
            spark.sql(insertQuery)

            labs = math.ceil(len(lasso_forward)/500.0)
            for i in range(labs):
                if (i + 1) * 500 < len(lasso_forward):
                    max_y = (i + 1) * 500
                else:
                    max_y = len(lasso_forward)

                insertQuery = "Insert into bizanal.fdcalysis_coeff VALUES"
                for idx, row in lasso_forward[i * 500:max_y].iterrows():
                    insertQuery += create_insert_value(result_code,
                                                       row['Variable'], row['Score'], row['ABS_Score'],
                                                       job_id, job_id[0:8]) + ","

                insertQuery = insertQuery[0:len(insertQuery)-1]
                spark.sql(insertQuery)

                laps = math.ceil(len(rf_and_lasso_feature_corr_final) / 500.0)
                for i in range(laps):
                    if(i + 1) * 500 < len(rf_and_lasso_feature_corr_final):
                        max_y = (i + 1) * 500
                    else:
                        max_y = len(rf_and_lasso_feature_corr_final)

                    insertQuery = "Insert into bizanal.fdcanalysis_final VALUES"
                    for idx, row in rf_and_lasso_feature_corr_final[i*500:max_y].iterrows():
                        insertQuery += create_insert_value('0',
                                                  row['Final'], row['Suggestion'],
                                                  row['Absolute Spearman Corr.(Final-Suggestion)'],
                                                  row['Absolute Spearman Corr.(Target-Suggestion)'],
                                                  row['Maximal Information Coeff.(Target-Suggestion)'],
                                                  row['mean_of_corr'],
                                                  row['mean_of_corr(mic)'],
                                                  job_id, job_id[0:8]) + ","
                    insertQuery =  insertQuery[0:len(insertQuery)-1]
                    spark.sql(insertQuery)

            total_elapsed = time.time() - total_time
            print(f"[INFO] Regression analysis completed successfully in {total_elapsed:.2f}s")
            print(f"[INFO] Result code: {result_code}")                              
                        
                        
                


                                                     
                                                     
                                    
                

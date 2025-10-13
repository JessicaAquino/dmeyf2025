import logging

import src.config.conf as cf
import src.infra.loader_utils as lu
import src.core.col_selection as cs
import src.core.feature_engineering as fe
import src.core.preprocessing as pp
import src.config.logger_config as lc
import src.ml.lgbm_train_test as tt


import optuna
import json
import polars as pl

logger = logging.getLogger(__name__)

# region Config_vars

cfg = cf.load_config("CHALLENGE_01")
paths = cfg.get('PATHS', None)

# Experiment values
STUDY_NAME = cfg.get('STUDY_NAME', None)

SEEDS = cfg.get('SEEDS', None)
TOP_N = cfg.get('TOP_N')

MONTH_TRAIN = cfg.get('MONTH_TRAIN', None)
MONTH_VALIDATION = cfg.get('MONTH_VALIDATION', None)
MONTH_TEST = cfg.get('MONTH_TEST', None)

GAIN_AMOUNT = cfg.get('GAIN')
COST_AMOUNT = cfg.get('COST')


BINARY_POSITIVES = cfg.get('BINARY_POSITIVES', None)

LGBM_N_TRIALS = cfg.get('LGBM_N_TRIALS', None)
LGBM_N_FOLDS = cfg.get('LGBM_N_FOLDS', None)
LGBM_N_BOOSTS = cfg.get('LGBM_N_BOOSTS', None)
LGBM_THRESHOLD = cfg.get('LGBM_THRESHOLD', None)

# Paths

## Logs
PATH_LOGS = paths.get('LOGS', None)

## Input
PATH_DATA = paths.get('INPUT_DATA', None)

## Output
PATH_LGBM_OPT = paths.get('OUTPUT_LGBM_OPTIMIZATION', None)
PATH_LGBM_OPT_BEST_PARAMS = paths.get('OUTPUT_LGBM_OPTIMIZATION_BEST_PARAMS', None)
PATH_LGBM_OPT_DB = paths.get('OUTPUT_LGBM_OPTIMIZATION_DB', None)

PATH_LGBM_MODEL = paths.get('OUTPUT_LGBM_MODEL', None)

PATH_PREDICTION = paths.get('OUTPUT_PREDICTION')

PATH_GRAPHICS = paths.get('OUTPUT_GRAPHICS')

# endregion 

def main():
    # Nombre de los hiperparametros a utilizar en conf.yaml
    # STUDY_NAME = "_20251003"
    
    NEW_STUDY = "_20251012_02"

    logger.info("STARTING this wonderful pipeline!")

    # 0. Load data
    df = lu.load_data(f"{PATH_DATA}competencia_01.csv", "csv")

    # 1. Columns selection
    cols_lag_delta_max_min_regl, cols_ratios = cs.col_selection(df)

    # 2. Feature Engineering
    df = fe.feature_engineering_pipeline(df, {
        "lag": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        "delta": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        "ratio": {
            "pairs": cols_ratios
        },
    })

    # 3. Preprocessing
    MONTH_TRAIN.append(MONTH_VALIDATION)

    X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test = pp.preprocessing_pipeline(
        df,
        BINARY_POSITIVES,
        MONTH_TRAIN,
        MONTH_TEST
    )

    # 4. Best hyperparams loading
    name_best_params_file = f"best_params_binary{STUDY_NAME}.json"
    storage_name = "sqlite:///" + PATH_LGBM_OPT_DB + "optimization_lgbm_best.db"
    study = optuna.load_study(study_name='study_lgbm_binary'+STUDY_NAME, storage=storage_name)
    
    # 5. Training with best attempt and hyperparams
    best_iter = study.best_trial.user_attrs["best_iter"]
    
    with open(PATH_LGBM_OPT_BEST_PARAMS + name_best_params_file, "r") as f:
        best_params = json.load(f)
    logger.info(f"Hyperparams OK?: {study.best_trial.params == best_params}")

    tt_cfg = []

    for i in range(0, 5):
        cfg = tt.TrainTestConfig(
            gain_amount=GAIN_AMOUNT,
            cost_amount=COST_AMOUNT,

            name=NEW_STUDY+"_0"+str(i),

            output_path=PATH_LGBM_MODEL,
            seeds=[SEEDS[i]]
        )

        tt_cfg.append(cfg)

    y_test_binary = X_test[["numero_de_cliente"]].copy()

    for i in range(0, 5):
        model = tt.entrenamiento_lgbm(X_train, y_train_binary, w_train ,best_iter,best_params , tt_cfg[i])
        y_test_binary["PredictedProb_0"+str(i)] = model.predict(X_test)

    y_test_binary["PredictedProb"] = y_test_binary.filter(like="PredictedProb_").mean(axis=1)
    
    # Sort descending by predicted probability
    y_test_binary = y_test_binary.sort_values("PredictedProb", ascending=False)

    # Assign 1 to top N, 0 to the rest
    y_test_binary["Predicted"] = 0
    y_test_binary.iloc[:TOP_N, y_test_binary.columns.get_loc("Predicted")] = 1

    y_test_binary_copy = y_test_binary.set_index("numero_de_cliente")
    y_test_binary_copy.to_csv(f"output/prediction/prediccion{NEW_STUDY}_prob.csv")

    y_test_binary = y_test_binary.set_index("numero_de_cliente")[["Predicted"]]
    y_test_binary.to_csv(f"output/prediction/prediccion{NEW_STUDY}.csv")

    logger.info("Pipeline ENDED!")

def get_top_n_predictions(csv_path: str, n: int) -> pl.DataFrame:

    # 1. Read CSV
    df = pl.read_csv(csv_path)

    # 2. Sort descending by PredictedProb
    df = df.sort("PredictedProb_01", descending=True)

    # 3. Assign 1 to top N, 0 to rest
    df = df.with_columns(
        pl.when(pl.arange(0, df.height) < n)
        .then(1)
        .otherwise(0)
        .alias("Predicted")
    )

    # 4. Keep only the two columns
    return df.select(["numero_de_cliente", "Predicted"])

if __name__ == "__main__":
    lu.ensure_dirs(
        PATH_LOGS,
        # PATH_DATA,
        PATH_LGBM_OPT,
        PATH_LGBM_OPT_BEST_PARAMS,
        PATH_LGBM_OPT_DB,
        PATH_LGBM_MODEL,
        PATH_PREDICTION,
        PATH_GRAPHICS
    )
    lc.setup_logging(PATH_LOGS)

    main()

    # Del archivo resultante de ensamblar multiples semillas, 
    # tomo los primeros 10500 de la segunda semilla 
    top_clients = get_top_n_predictions("output/prediction/prediccion_20251012_02_prob.csv", n=TOP_N)

    top_clients.write_csv("output/prediction/prediccion_20251012_18.csv")

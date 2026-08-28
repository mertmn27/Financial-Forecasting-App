import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred) / y_true
        )
    ) * 100

    return mae, rmse, r2, mape


def train_model(
        model_name,
        X_train,
        y_train,
        X_test,
        y_test,
        store_column=None,
        n_estimators=None,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        learning_rate=None,
        subsample=None,
        lookback=4,
        epochs=50,
        batch_size=16
):

    test_store = None

    if store_column is not None and store_column in X_test.columns:
        test_store = X_test[store_column].copy()

    # ========================================================
    # LSTM
    # ========================================================

    if model_name == "LSTM":

        X_train = X_train.copy()
        X_test = X_test.copy()

        y_train = y_train.copy()
        y_test = y_test.copy()

        # ----------------------------------------------------
        # MAĞAZA BİLGİSİ VARSA
        # ----------------------------------------------------

        if store_column is not None:

            if store_column not in X_train.columns:
                raise ValueError(
                    f"'{store_column}' sütunu Train verisinde bulunamadı."
                )

            if store_column not in X_test.columns:
                raise ValueError(
                    f"'{store_column}' sütunu Test verisinde bulunamadı."
                )

            train_store = X_train[store_column].to_numpy()
            test_store = X_test[store_column].to_numpy()

            X_train_features = X_train.drop(
                columns=[store_column]
            )

            X_test_features = X_test.drop(
                columns=[store_column]
            )

        # ----------------------------------------------------
        # MAĞAZA BİLGİSİ YOKSA
        # ----------------------------------------------------

        else:

            train_store = None
            test_store = None

            X_train_features = X_train.copy()
            X_test_features = X_test.copy()

        # ----------------------------------------------------
        # TRAIN / TEST SÜTUNLARINI EŞİTLE
        # ----------------------------------------------------

        X_test_features = X_test_features.reindex(
            columns=X_train_features.columns,
            fill_value=0
        )

        # ----------------------------------------------------
        # FEATURE SCALER
        # ----------------------------------------------------

        feature_scaler = StandardScaler()

        X_train_scaled = feature_scaler.fit_transform(
            X_train_features
        ).astype(np.float32)

        X_test_scaled = feature_scaler.transform(
            X_test_features
        ).astype(np.float32)

        # ----------------------------------------------------
        # TARGET SCALER
        # ----------------------------------------------------

        target_scaler = StandardScaler()

        y_train_scaled = target_scaler.fit_transform(
            y_train.to_numpy().reshape(-1, 1)
        ).flatten().astype(np.float32)

        y_test_values = y_test.to_numpy()

        # ====================================================
        # TRAIN SEQUENCE OLUŞTURMA
        # ====================================================

        X_train_sequences = []
        y_train_sequences = []

        # ====================================================
        # MAĞAZA VARSA → MAĞAZA BAZLI SEQUENCE
        # ====================================================

        if train_store is not None:

            unique_stores = np.unique(
                train_store
            )

            for store in unique_stores:

                train_mask = (
                    train_store == store
                )

                store_X = X_train_scaled[
                    train_mask
                ]

                store_y = y_train_scaled[
                    train_mask
                ]

                if len(store_X) <= lookback:
                    continue

                store_sequence_data = np.concatenate(
                    [
                        store_X,
                        store_y.reshape(-1, 1)
                    ],
                    axis=1
                )

                for i in range(
                    lookback,
                    len(store_sequence_data)
                ):

                    X_train_sequences.append(
                        store_sequence_data[
                            i - lookback:i
                        ]
                    )

                    y_train_sequences.append(
                        store_y[i]
                    )

        # ====================================================
        # MAĞAZA YOKSA → TEK ZAMAN SERİSİ
        # ====================================================

        else:

            if len(X_train_scaled) <= lookback:

                raise ValueError(
                    "LSTM için yeterli train verisi bulunamadı. "
                    "Lookback değerini azaltın."
                )

            train_sequence_data = np.concatenate(
                [
                    X_train_scaled,
                    y_train_scaled.reshape(-1, 1)
                ],
                axis=1
            )

            for i in range(
                lookback,
                len(train_sequence_data)
            ):

                X_train_sequences.append(
                    train_sequence_data[
                        i - lookback:i
                    ]
                )

                y_train_sequences.append(
                    y_train_scaled[i]
                )

        # ====================================================
        # NUMPY ARRAY
        # ====================================================

        X_train_sequences = np.asarray(
            X_train_sequences,
            dtype=np.float32
        )

        y_train_sequences = np.asarray(
            y_train_sequences,
            dtype=np.float32
        )

        # ====================================================
        # TRAIN KONTROL
        # ====================================================

        if len(X_train_sequences) == 0:

            raise ValueError(
                "LSTM için yeterli train verisi bulunamadı. "
                "Lookback değerini azaltın."
            )

        # ====================================================
        # LSTM MODELİ
        # ====================================================

        model = Sequential(
            [
                Input(
                    shape=(
                        X_train_sequences.shape[1],
                        X_train_sequences.shape[2]
                    )
                ),

                LSTM(
                    32
                ),

                Dense(
                    1
                )
            ]
        )

        model.compile(
            optimizer="adam",
            loss="mse"
        )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        early_stopping = EarlyStopping(
            monitor="loss",
            patience=3,
            restore_best_weights=True
        )

        # ====================================================
        # MODEL EĞİTİMİ
        # ====================================================

        model.fit(
            X_train_sequences,
            y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping]
        )

        # ====================================================
        # TAHMİN FONKSİYONU
        # ====================================================

        @tf.function(
            reduce_retracing=True
        )
        def predict_sequence(sequence):

            return model(
                sequence,
                training=False
            )

        # ====================================================
        # TEST
        # ====================================================

        test_predictions = []
        test_actuals = []
        test_sequence_stores = []

        # ====================================================
        # MAĞAZA VARSA → MAĞAZA BAZLI TAHMİN
        # ====================================================

        if test_store is not None:

            unique_test_stores = np.unique(
                test_store
            )

            for store in unique_test_stores:

                train_mask = (
                    train_store == store
                )

                store_train_X = X_train_scaled[
                    train_mask
                ]

                store_train_y = y_train_scaled[
                    train_mask
                ]

                test_mask = (
                    test_store == store
                )

                store_test_X = X_test_scaled[
                    test_mask
                ]

                store_test_y = y_test_values[
                    test_mask
                ]

                if len(store_train_X) < lookback:
                    continue

                if len(store_test_X) == 0:
                    continue

                history_X = np.empty(
                    (
                        lookback + len(store_test_X),
                        store_train_X.shape[1]
                    ),
                    dtype=np.float32
                )

                history_y = np.empty(
                    lookback + len(store_test_X),
                    dtype=np.float32
                )

                history_X[:lookback] = (
                    store_train_X[-lookback:]
                )

                history_y[:lookback] = (
                    store_train_y[-lookback:]
                )

                for i in range(
                    len(store_test_X)
                ):

                    current_X = store_test_X[i]

                    sequence = np.empty(
                        (
                            1,
                            lookback,
                            store_train_X.shape[1] + 1
                        ),
                        dtype=np.float32
                    )

                    sequence[0, :, :-1] = (
                        history_X[i:i + lookback]
                    )

                    sequence[0, :, -1] = (
                        history_y[i:i + lookback]
                    )

                    sequence_tensor = tf.convert_to_tensor(
                        sequence
                    )

                    prediction_scaled = (
                        predict_sequence(
                            sequence_tensor
                        )
                        .numpy()[0, 0]
                    )

                    prediction = (
                        prediction_scaled
                        * target_scaler.scale_[0]
                        + target_scaler.mean_[0]
                    )

                    test_predictions.append(
                        prediction
                    )

                    test_actuals.append(
                        store_test_y[i]
                    )

                    test_sequence_stores.append(
                        store
                    )

                    history_X[
                        i + lookback
                    ] = current_X

                    history_y[
                        i + lookback
                    ] = prediction_scaled

        # ====================================================
        # MAĞAZA YOKSA → TEK ZAMAN SERİSİ TAHMİNİ
        # ====================================================

        else:

            if len(X_train_scaled) < lookback:
                raise ValueError(
                    "LSTM için yeterli train verisi bulunamadı."
                )

            history_X = np.empty(
                (
                    lookback + len(X_test_scaled),
                    X_train_scaled.shape[1]
                ),
                dtype=np.float32
            )

            history_y = np.empty(
                lookback + len(X_test_scaled),
                dtype=np.float32
            )

            history_X[:lookback] = (
                X_train_scaled[-lookback:]
            )

            history_y[:lookback] = (
                y_train_scaled[-lookback:]
            )

            for i in range(
                len(X_test_scaled)
            ):

                current_X = X_test_scaled[i]

                sequence = np.empty(
                    (
                        1,
                        lookback,
                        X_train_scaled.shape[1] + 1
                    ),
                    dtype=np.float32
                )

                sequence[0, :, :-1] = (
                    history_X[i:i + lookback]
                )

                sequence[0, :, -1] = (
                    history_y[i:i + lookback]
                )

                sequence_tensor = tf.convert_to_tensor(
                    sequence
                )

                prediction_scaled = (
                    predict_sequence(
                        sequence_tensor
                    )
                    .numpy()[0, 0]
                )

                prediction = (
                    prediction_scaled
                    * target_scaler.scale_[0]
                    + target_scaler.mean_[0]
                )

                test_predictions.append(
                    prediction
                )

                test_actuals.append(
                    y_test_values[i]
                )

                # Mağaza olmadığı için None kullanıyoruz
                test_sequence_stores.append(
                    None
                )

                history_X[
                    i + lookback
                ] = current_X

                history_y[
                    i + lookback
                ] = prediction_scaled

        # ====================================================
        # NUMPY / SERIES
        # ====================================================

        y_pred = np.asarray(
            test_predictions
        )

        y_test_final = pd.Series(
            test_actuals
        )

        test_sequence_stores = pd.Series(
            test_sequence_stores
        )

        # ====================================================
        # TEST KONTROL
        # ====================================================

        if len(y_pred) == 0:

            raise ValueError(
                "LSTM için yeterli test verisi bulunamadı."
            )

        # ====================================================
        # METRİKLER
        # ====================================================

        mae, rmse, r2, mape = calculate_metrics(
            y_test_final,
            y_pred
        )

        # ====================================================
        # SONUÇ
        # ====================================================

        return {

            "model": model,

            "y_pred": y_pred,

            "y_test": y_test_final,

            "mae": mae,

            "rmse": rmse,

            "r2": r2,

            "mape": mape,

            "test_store": test_sequence_stores,

            "scaler": feature_scaler,

            "target_scaler": target_scaler,

            "lookback": lookback
        }



    # ========================================================
    # DİĞER MODELLER
    # ========================================================

    if store_column is not None and store_column in X_train.columns:

        X_train = pd.get_dummies(
            X_train,
            columns=[store_column],
            drop_first=True,
            dtype=int
        )

        X_test = pd.get_dummies(
            X_test,
            columns=[store_column],
            drop_first=True,
            dtype=int
        )

        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0
        )

    # ========================================================
    # MODEL OLUŞTUR
    # ========================================================

    if model_name == "Linear Regression":

        model = LinearRegression()

    elif model_name == "Random Forest":

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "XGBoost":

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
            n_jobs=-1
        )

    else:

        raise ValueError(
            f"Desteklenmeyen model: {model_name}"
        )

    # ========================================================
    # MODEL EĞİTİMİ
    # ========================================================

    model.fit(
        X_train,
        y_train
    )

    # ========================================================
    # TAHMİN
    # ========================================================

    y_pred = model.predict(
        X_test
    )

    # ========================================================
    # METRİKLER
    # ========================================================

    mae, rmse, r2, mape = calculate_metrics(
        y_test,
        y_pred
    )

    # ========================================================
    # SONUÇ
    # ========================================================

    return {

        "model": model,

        "y_pred": y_pred,

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

        "mape": mape,

        "test_store": test_store,

        "y_test": y_test.copy()
    }
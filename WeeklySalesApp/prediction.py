import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def run_prediction(
    df,
    forecast_store,
    forecast_horizon
):

    # ========================================================
    # DATE → YEAR / MONTH / WEEK
    # ========================================================

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%Y-%m-%d"
    )

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["Week"] = (
        df["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )


    # ========================================================
    # WEEK SIN / WEEK COS
    # ========================================================

    df["Week_Sin"] = np.sin(
        2 * np.pi * df["Week"] / 52
    )

    df["Week_Cos"] = np.cos(
        2 * np.pi * df["Week"] / 52
    )


    # ========================================================
    # SEÇİLEN MAĞAZALAR
    # ========================================================

    selected_stores = [
        31, 4, 34, 32, 13,
        45, 44, 37, 25, 22,
        30, 43, 11, 1, 8,
        2, 20, 6, 19, 18
    ]


    # ========================================================
    # SADECE SEÇİLEN MAĞAZALAR
    # ========================================================

    df_model = df[
        df["Store"].isin(selected_stores)
    ].copy()


    # ========================================================
    # GEREKLİ SÜTUNLAR
    # ========================================================

    required_columns = [
        "Store",
        "Date",
        "Year",
        "Month",
        "Week",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Average_Wage",
        "Retail_Sales_Growth",
        "Week_Sin",
        "Week_Cos",
        "Weekly_Sales"
    ]

    for column in required_columns:

        if column not in df_model.columns:

            raise ValueError(
                f"{column} sütunu veri setinde bulunamadı."
            )


    # ========================================================
    # BENZERSİZ HAFTALAR
    # ========================================================

    week_table = (
        df_model[
            [
                "Year",
                "Month",
                "Week"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Year",
                "Month",
                "Week"
            ]
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # TIME INDEX
    # ========================================================

    week_table["Time_Index"] = np.arange(
        1,
        len(week_table) + 1
    )


    # ========================================================
    # TIME INDEX'İ ANA VERİYE EKLE
    # ========================================================

    df_model = df_model.merge(
        week_table,
        on=[
            "Year",
            "Month",
            "Week"
        ],
        how="left"
    )


    # ========================================================
    # SIRALAMA
    # ========================================================

    df_model = df_model.sort_values(
        [
            "Store",
            "Time_Index"
        ]
    ).reset_index(drop=True)


    # ========================================================
    # AYIN İLK VE SON HAFTASINI BUL
    # ========================================================

    month_week_info = (
        df_model
        .groupby(
            [
                "Year",
                "Month"
            ]
        )["Week"]
        .agg(
            First_Week="min",
            Last_Week="max"
        )
        .reset_index()
    )


    # ========================================================
    # BİLGİLERİ ANA VERİYE EKLE
    # ========================================================

    df_model = df_model.merge(
        month_week_info,
        on=[
            "Year",
            "Month"
        ],
        how="left"
    )


    # ========================================================
    # AYIN İLK HAFTASI
    # ========================================================

    df_model["Is_Month_Start"] = (
        df_model["Week"]
        == df_model["First_Week"]
    ).astype(int)


    # ========================================================
    # AYIN SON HAFTASI
    # ========================================================

    df_model["Is_Month_End"] = (
        df_model["Week"]
        == df_model["Last_Week"]
    ).astype(int)


    # ========================================================
    # YARDIMCI SÜTUNLARI KALDIR
    # ========================================================

    df_model = df_model.drop(
        columns=[
            "First_Week",
            "Last_Week"
        ]
    )


    # ========================================================
    # MODEL FEATURE'LARI
    # ========================================================

    features = [
        "Store",
        "Year",
        "Month",
        "Week",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Average_Wage",
        "Retail_Sales_Growth",
        "Week_Sin",
        "Week_Cos",
        "Is_Month_Start",
        "Is_Month_End"
    ]


    # ========================================================
    # SON HAFTAYI BUL
    # ========================================================

    last_week = df_model[
        "Time_Index"
    ].max()


    # ========================================================
    # TAHMİN EDİLECEK HAFTALAR
    # ========================================================

    forecast_weeks = list(
        range(
            last_week - forecast_horizon + 1,
            last_week + 1
        )
    )


    # ========================================================
    # TRAIN'İN SON HAFTASI
    # ========================================================

    train_last_week = (
        last_week - forecast_horizon
    )


    # ========================================================
    # TRAIN
    # ========================================================

    train = df_model[
        df_model["Time_Index"]
        <= train_last_week
    ].copy()


    # ========================================================
    # TEST
    # ========================================================

    test = df_model[
    (df_model["Time_Index"].isin(forecast_weeks)) &
    (df_model["Store"] == forecast_store)
    ].copy()


    # ========================================================
    # X VE y
    # ========================================================

    X_train = train[
        features
    ].copy()

    y_train = train[
        "Weekly_Sales"
    ].copy()

    X_test = test[
        features
    ].copy()

    y_test = test[
        "Weekly_Sales"
    ].copy()


    # ========================================================
    # STORE DUMMY
    # TRAIN + TEST BİRLİKTE
    # ========================================================

    X_combined = pd.concat(
        [
            X_train,
            X_test
        ],
        axis=0
    )

    X_combined = pd.get_dummies(
        X_combined,
        columns=["Store"],
        drop_first=True,
        dtype=int
    )


    # ========================================================
    # TRAIN / TEST OLARAK GERİ AYIR
    # ========================================================

    X_train = X_combined.iloc[
        :len(X_train)
    ].copy()

    X_test = X_combined.iloc[
        len(X_train):
    ].copy()


    # ========================================================
    # SÜTUNLARI EŞİTLE
    # ========================================================

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )


    # ========================================================
    # MODELİ EĞİT
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

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    mape = np.mean(
        np.abs(
            (y_test - y_pred)
            / y_test
        )
    ) * 100


    # ========================================================
    # SONUÇ TABLOSU
    # ========================================================

    test_results = test[
        [
            "Store",
            "Date",
            "Weekly_Sales"
        ]
    ].copy()

    test_results["Predicted_Sales"] = y_pred

    test_results = test_results.rename(
        columns={
            "Weekly_Sales": "Actual_Sales"
        }
    )


    # ========================================================
    # SADECE SEÇİLEN MAĞAZANIN SONUÇLARI
    # ========================================================

    store_results = test_results[
        test_results["Store"] == forecast_store
    ].copy()

    # ========================================================
# TÜM MAĞAZALAR İÇİN ORTALAMA METRİKLER
# ========================================================

    all_metrics = []

    for store in test_results["Store"].unique():

        store_data = test_results[
            test_results["Store"] == store
        ]

        y_true_store = store_data[
            "Actual_Sales"
        ]

        y_pred_store = store_data[
            "Predicted_Sales"
        ]

        store_mae = mean_absolute_error(
            y_true_store,
            y_pred_store
        )

        store_rmse = np.sqrt(
            mean_squared_error(
                y_true_store,
                y_pred_store
            )
        )

        store_r2 = r2_score(
            y_true_store,
            y_pred_store
        )

        store_mape = np.mean(
            np.abs(
                (y_true_store - y_pred_store)
                / y_true_store
            )
        ) * 100

        all_metrics.append({
            "MAE": store_mae,
            "RMSE": store_rmse,
            "R2": store_r2,
            "MAPE": store_mape
        })


    average_metrics = pd.DataFrame(
        all_metrics
    ).mean()

    # ========================================================
    # SONUÇLARI GERİ DÖNDÜR
    # ========================================================

    return {
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "mape": mape,

    "results": store_results,
    "all_results": test_results,

    "average_metrics": average_metrics,

    "last_week": last_week,
    "train_last_week": train_last_week,
    "forecast_weeks": forecast_weeks,

    "model": model
}
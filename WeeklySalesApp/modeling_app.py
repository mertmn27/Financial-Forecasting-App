import json
import streamlit as st
import pandas as pd
import numpy as np

from mrmr import mrmr_classif
from mrmr import mrmr_regression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from models import train_model


def create_date_features(df, date_column):

    df = df.copy()

    if date_column not in df.columns:
        return 

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["Year"] = df[date_column].dt.year

    df["Month"] = df[date_column].dt.month

    df["Week"] = (
        df[date_column]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["Week_Sin"] = np.sin(
        2 * np.pi * df["Week"] / 52
    )

    df["Week_Cos"] = np.cos(
        2 * np.pi * df["Week"] / 52
    )

    month_week_info = (
        df
        .groupby(
            ["Year", "Month"]
        )["Week"]
        .agg(
            First_Week="min",
            Last_Week="max"
        )
        .reset_index()
    )

    df = df.merge(
        month_week_info,
        on=["Year", "Month"],
        how="left"
    )

    df["Is_Month_Start"] = (
        df["Week"] == df["First_Week"]
    ).astype(int)

    df["Is_Month_End"] = (
        df["Week"] == df["Last_Week"]
    ).astype(int)

    df = df.drop(
        columns=[
            "First_Week",
            "Last_Week"
        ]
    )

    return df


# ========================================================
# SAYFA AYARLARI
# ========================================================

st.set_page_config(
    page_title="Modelleme Paneli",
    page_icon="📊",
    layout="wide"
)

# ========================================================
# BAŞLIK
# ========================================================

st.markdown(
    "<h1 style='text-align:center;'>Modelleme ve Tahmin Paneli</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; '> Veri setlerini yükleyin, öznitelikleri seçin ve modelinizi oluşturun</p>",
    unsafe_allow_html=True
)

st.divider()


# ========================================================
# KAYITLI MODEL SONUCU YÜKLE
# ========================================================

st.markdown(
    "<h2 style='text-align:center;'>📂 Kayıtlı Sonuç Yükle</h2>",
    unsafe_allow_html=True
)

result_file = st.file_uploader(
    "Daha önce kaydettiğiniz JSON sonucunu yükleyin:",
    type=["json"],
    key="result_json"
)


# ========================================================
# JSON YÜKLENDİYSE
# ========================================================

if result_file is not None:

    loaded_result = json.load(result_file)

    st.session_state["result_source"] = "json"

    st.session_state["model_result"] = {

        "y_pred": np.array(
            loaded_result["y_pred"]
        ),

        "y_test": pd.Series(
            loaded_result["y_test"]
        ),

        "mae": float(
            loaded_result["mae"]
        ),

        "rmse": float(
            loaded_result["rmse"]
        ),

        "r2": float(
            loaded_result["r2"]
        ),

        "mape": float(
            loaded_result["mape"]
        ),

        "test_store": pd.Series(
            loaded_result["test_store"]
        )
    }

    st.session_state["loaded_model_name"] = (
        loaded_result["model_name"]
    )

    st.session_state["loaded_features"] = (
        loaded_result.get(
            "selected_features",
            []
        )
    )
    

    st.success(
        f"{loaded_result['model_name']} sonucu başarıyla yüklendi."
    )

    # ====================================================
    # YÜKLENEN MODEL BİLGİLERİ
    # ====================================================

    st.divider()

    st.markdown(
        "<h2 style='text-align:center;'>⚙️ Yüklenen Model Bilgileri</h2>",
        unsafe_allow_html=True
    )

    loaded_model_name = loaded_result["model_name"]

    # ====================================================
    # MODEL BİLGİLERİ
    # ====================================================

    if loaded_model_name == "LSTM":

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Model",
                loaded_model_name
            )

        with col2:
            st.metric(
                "Lookback",
                loaded_result.get("lookback", "-")
            )

        with col3:
            st.metric(
                "Epoch",
                loaded_result.get("epochs", "-")
            )

        with col4:
            st.metric(
                "Batch Size",
                loaded_result.get("batch_size", "-")
            )


    elif loaded_model_name == "Random Forest":

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Model",
                loaded_model_name
            )

        with col2:
            st.metric(
                "n_estimators",
                loaded_result.get("n_estimators", "-")
            )

        with col3:
            st.metric(
                "max_depth",
                loaded_result.get("max_depth", "-")
            )

        with col4:
            st.metric(
                "min_samples_split",
                loaded_result.get("min_samples_split", "-")
            )

        with col5:
            st.metric(
                "min_samples_leaf",
                loaded_result.get("min_samples_leaf", "-")
            )


    elif loaded_model_name == "XGBoost":

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Model",
                loaded_model_name
            )

        with col2:
            st.metric(
                "n_estimators",
                loaded_result.get("n_estimators", "-")
            )

        with col3:
            st.metric(
                "max_depth",
                loaded_result.get("max_depth", "-")
            )

        with col4:
            st.metric(
                "Learning Rate",
                loaded_result.get("learning_rate", "-")
            )

        with col5:
            st.metric(
                "Subsample",
                loaded_result.get("subsample", "-")
            )


    elif loaded_model_name == "Linear Regression":

        col1 = st.columns(1)[0]

        with col1:
            st.metric(
                "Model",
                loaded_model_name
            )

    st.divider()

    # ====================================================
    # KULLANILAN ÖZNİTELİKLER
    # ====================================================

    st.markdown("### 🔎 Kullanılan Öznitelikler")

    st.write(
        "///".join(
            loaded_result.get(
                "selected_features",
                []
            )
        )
    )

else:
    # ====================================================
    # JSON KALDIRILDIYSA ESKİ SONUÇLARI TEMİZLE
    # ====================================================

    if st.session_state.get("result_source") == "json":

        st.session_state.pop(
            "model_result",
            None
        )

        st.session_state.pop(
            "loaded_model_name",
            None
        )

        st.session_state.pop(
            "loaded_features",
            None
        )

        st.session_state.pop(
            "result_source",
            None
        )

        st.rerun()

    # ====================================================
    # TRAIN / TEST CSV
    # ====================================================

    st.markdown(
        "<h2 style='text-align:center;'>📂 Veri Setleri</h2>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        train_file = st.file_uploader(
            "Train CSV dosyasını yükleyin",
            type=["csv"],
            key="train_file"
        )

    with col2:

        test_file = st.file_uploader(
            "Test CSV dosyasını yükleyin",
            type=["csv"],
            key="test_file"
        )

    # ====================================================
    # DOSYALAR YÜKLENDİYSE
    # ====================================================

    if train_file is not None and test_file is not None:

        train_df = pd.read_csv(train_file)

        test_df = pd.read_csv(test_file)

        st.success(
            "Train ve Test dosyaları başarıyla yüklendi."
        )

        # ========================================================
        # TARİH BİLGİSİ
        # ========================================================

        date_column_valid = False
        
        st.subheader("📅 Tarih Bilgisi")

        date_column = st.text_input(
            "Tarih bilgisini içeren sütunun adını giriniz:",
            placeholder="Örn: Date, Tarih, Transaction_Date"
        ).strip()

        date_column_valid = True

        if date_column == "":

            date_column_valid = False

        elif date_column not in train_df.columns:

            date_column_valid = False

            st.error(
                f"❌ Train CSV içerisinde "
                f"'{date_column}' sütunu bulunamadı."
            )

        elif date_column not in test_df.columns:

            date_column_valid = False

            st.error(
                f"❌ Test CSV içerisinde "
                f"'{date_column}' sütunu bulunamadı."
            )

        else:

            st.success(
                f"✅ '{date_column}' tarih sütunu olarak kullanılacak."
            )
        # MAĞAZA BİLGİSİ

        st.subheader("🏪 Mağaza / Şube Bilgisi")

        has_store = st.radio(
            "Veri setinizde birbirinden bağımsız mağaza veya şubeler var mı?",
            ["Evet","Hayır"],
            horizontal=True
        )

        store_column = None
        store_column_input = ""
        store_column_valid = True

        if has_store == "Evet":

            store_column_input = st.text_input(
                "Mağazaları belirten sütunun adını giriniz:",
                placeholder="Örn: Store, Store_ID, Branch, Branch_ID"
            ).strip()

            if store_column_input:

                if store_column_input == "":
                    store_column_valid = False

                elif store_column_input not in train_df.columns:

                    store_column_valid = False

                    st.error(
                        f"❌ Train CSV içerisinde '{store_column_input}' sütunu bulunamadı."
                    )

                elif store_column_input not in test_df.columns:

                    store_column_valid = False

                    st.error(
                        f"❌ Test CSV içerisinde '{store_column_input}' sütunu bulunamadı."
                    )


                else:

                    store_column = store_column_input

                    train_store_count = train_df[
                        store_column
                    ].nunique()

                    test_store_count = test_df[
                        store_column
                    ].nunique()

                    st.success(
                        f"✅ '{store_column}' sütunu bulundu. "
                        f"Train: {train_store_count} farklı mağaza, "
                        f"Test: {test_store_count} farklı mağaza."
                    )

        # ========================================================
        # TARİH SÜTUNU GEÇERLİYSE VERİYİ İŞLE
        # ========================================================

        if date_column_valid:

            train_df = create_date_features(
                train_df,
                date_column
            )

            test_df = create_date_features(
                test_df,
                date_column
            )

            # ====================================================
            # SIRALAMA
            # ====================================================

            if store_column is not None:

                train_df = train_df.sort_values(
                    [store_column, date_column]
                ).reset_index(drop=True)

                test_df = test_df.sort_values(
                    [store_column, date_column]
                ).reset_index(drop=True)

            else:

                train_df = train_df.sort_values(
                    date_column
                ).reset_index(drop=True)

                test_df = test_df.sort_values(
                    date_column
                ).reset_index(drop=True)

        # =================================================
        # VERİ SETİ BİLGİLERİ
        # =================================================

        st.markdown(
            "<h2 style='text-align:center;'>📊 Veri Seti Bilgileri</h2>",
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Train Satır Sayısı",
                f"{len(train_df):,}"
            )

        with col2:

            st.metric(
                "Train Sütun Sayısı",
                f"{len(train_df.columns):,}"
            )

        with col3:

            st.metric(
                "Test Satır Sayısı",
                f"{len(test_df):,}"
            )

        with col4:

            st.metric(
                "Test Sütun Sayısı",
                f"{len(test_df.columns):,}"
            )

        st.divider()

        # =================================================
        # HEDEF DEĞİŞKEN
        # =================================================

        st.subheader("🎯 Hedef Değişken")

        target_column = st.selectbox(
            "Tahmin edilecek sütunu seçin:",
            train_df.columns
        )

        # =================================================
        # ÖZNİTELİK SEÇİMİ
        # =================================================

        st.subheader("🔎 Öznitelik Seçimi")

        feature_selection_method = st.radio(
            "Öznitelik seçim yöntemini seçin:",
            [
                "Manuel Seçim",
                "mRMR"
            ],
            horizontal=True
        )

        if feature_selection_method == "Manuel Seçim":

            available_features = [
                column
                for column in train_df.columns
                if column != target_column
                and column != date_column
            ]

            selected_features = st.multiselect(
                "Modelde kullanılacak öznitelikleri seçin:",
                available_features
            )

        elif feature_selection_method == "mRMR":

            available_features = [
                column
                for column in train_df.columns
                if column != target_column
                and column != "Date"
            ]

            numeric_features = train_df[
                available_features
            ].select_dtypes(
                include=np.number
            ).columns.tolist()

            feature_count = st.number_input(
                f"Kaç öznitelik seçilsin? ({len(numeric_features)} öznitelik seçilebilir)",
                min_value=1,
                max_value=len(numeric_features),
                value=min(5, len(numeric_features)),
                step=1
            )

            if len(numeric_features) == 0:

                st.error(
                    "mRMR için kullanılabilecek sayısal öznitelik bulunamadı."
                )

                selected_features = []

            elif feature_count > len(numeric_features):

                st.warning(
                    f"En fazla {len(numeric_features)} sayısal öznitelik kullanılabilir."
                )

                selected_features = []

            else:

                X_mrmr = train_df[
                    numeric_features
                ].copy()

                y_mrmr = train_df[
                    target_column
                ].copy()

                selected_features = mrmr_regression(
                    X=X_mrmr,
                    y=y_mrmr,
                    K=feature_count,
                    n_jobs=1
                )

                st.success(
                    f"{len(selected_features)} öznitelik seçildi."
                )

                st.write(
                    "Seçilen öznitelikler:"
                )

                st.write(
                    selected_features
                )

        st.divider()

        # =================================================
        # MODEL SEÇİMİ
        # =================================================

        st.markdown(
            "<h2 style='text-align:center;'>⚙️ Model Ayarları</h2>",
            unsafe_allow_html=True
        )

        model_name = st.selectbox(
            "Kullanılacak modeli seçin:",
            [
                "Linear Regression",
                "Random Forest",
                "XGBoost",
                "LSTM"
            ]
        )

        # =================================================
        # RANDOM FOREST
        # =================================================

        if model_name == "Random Forest":

            col1, col2 = st.columns(2)

            with col1:

                n_estimators = st.number_input(
                    "n estimators (10-1000)",
                    min_value=10,
                    max_value=1000,
                    value=300,
                    step=10
                )

            with col2:

                max_depth = st.number_input(
                    "max_depth (1-100)",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1
                )

            col1, col2 = st.columns(2)

            with col1:

                min_samples_split = st.number_input(
                    "min_samples_split (2-20)",
                    min_value=2,
                    max_value=20,
                    value=2,
                    step=1
                )

            with col2:

                min_samples_leaf = st.number_input(
                    "min_samples_leaf (1-20)",
                    min_value=1,
                    max_value=20,
                    value=1,
                    step=1
                )

        # =================================================
        # LINEAR REGRESSION
        # =================================================

        elif model_name == "Linear Regression":

            st.info(
                "Linear Regression için ayarlanabilir temel hiperparametre bulunmamaktadır."
            )

        # =================================================
        # XGBOOST
        # =================================================

        elif model_name == "XGBoost":

            col1, col2 = st.columns(2)

            with col1:

                xgb_n_estimators = st.number_input(
                    "n_estimators (10-1000)",
                    min_value=10,
                    max_value=1000,
                    value=300,
                    step=10,
                    key="xgb_n_estimators"
                )

            with col2:

                xgb_learning_rate = st.number_input(
                    "learning rate (0.001-1.0)",
                    min_value=0.001,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    key="xgb_learning_rate"
                )

            col1, col2 = st.columns(2)

            with col1:

                xgb_max_depth = st.number_input(
                    "max_depth (1-20)",
                    min_value=1,
                    max_value=20,
                    value=6,
                    step=1,
                    key="xgb_max_depth"
                )

            with col2:

                xgb_subsample = st.number_input(
                    "subsample (0.1-1.0)",
                    min_value=0.1,
                    max_value=1.0,
                    value=1.0,
                    step=0.1,
                    key="xgb_subsample"
                )

        # =================================================
        # LSTM
        # =================================================

        elif model_name == "LSTM":

            col1, col2, col3 = st.columns(3)

            with col1:

                lookback = st.number_input(
                    "Lookback (geçmiş hafta) (1-20)",
                    min_value=1,
                    max_value=20,
                    value=4,
                    step=1
                )

            with col2:

                epochs = st.number_input(
                    "Epoch (1-200)",
                    min_value=1,
                    max_value=200,
                    value=50,
                    step=5
                )

            with col3:

                batch_size = st.number_input(
                    "Batch Size (1-128)",
                    min_value=1,
                    max_value=128,
                    value=32,
                    step=1
                )

        st.divider()

        # =================================================
        # MODELİ ÇALIŞTIR
        # =================================================

        run_model = st.button(
            "🚀 Modeli Çalıştır",
            type="primary",
            use_container_width=True
        )

        if run_model:

            if not date_column_valid:

                st.error(
                    "❌ Geçerli bir tarih sütunu girmeden model çalıştırılamaz."
                )

            elif has_store == "Evet" and not store_column_valid:

                st.error(
                    "❌ Geçerli bir mağaza sütunu girmeden model çalıştırılamaz."
                )

            elif len(selected_features) == 0:

                st.error(
                    "En az bir öznitelik seçmelisiniz."
                )

            else:

                model_features = selected_features.copy()

                if (
                    store_column is not None
                    and store_column not in model_features
                ):

                    model_features.append(
                        store_column
                    )
                
                X_train = train_df[
                    selected_features
                ].copy()

                y_train = train_df[
                    target_column
                ].copy()

                X_test = test_df[
                    selected_features
                ].copy()

                y_test = test_df[
                    target_column
                ].copy()

                result = train_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    store_column = store_column,

                    n_estimators=(
                        n_estimators
                        if model_name == "Random Forest"
                        else xgb_n_estimators
                        if model_name == "XGBoost"
                        else None
                    ),

                    max_depth=(
                        max_depth
                        if model_name == "Random Forest"
                        else xgb_max_depth
                        if model_name == "XGBoost"
                        else None
                    ),

                    min_samples_split=(
                        min_samples_split
                        if model_name == "Random Forest"
                        else None
                    ),

                    min_samples_leaf=(
                        min_samples_leaf
                        if model_name == "Random Forest"
                        else None
                    ),

                    learning_rate=(
                        xgb_learning_rate
                        if model_name == "XGBoost"
                        else None
                    ),

                    subsample=(
                        xgb_subsample
                        if model_name == "XGBoost"
                        else None
                    ),

                    lookback=(
                        lookback
                        if model_name == "LSTM"
                        else 4
                    ),

                    epochs=(
                        epochs
                        if model_name == "LSTM"
                        else 50
                    ),

                    batch_size=(
                        batch_size
                        if model_name == "LSTM"
                        else 16
                    )
                )

                st.session_state["model_result"] = result

                st.success(
                    f"{model_name} modeli başarıyla eğitildi."
                )

                # =================================================
                # JSON KAYDET
                # =================================================

                save_result = {

                    "model_name": model_name,

                    "mae": float(
                        result["mae"]
                    ),

                    "rmse": float(
                        result["rmse"]
                    ),

                    "r2": float(
                        result["r2"]
                    ),

                    "mape": float(
                        result["mape"]
                    ),

                    "y_pred": np.asarray(
                        result["y_pred"]
                    ).tolist(),

                    "y_test": np.asarray(
                        result["y_test"]
                    ).tolist(),

                    "test_store": np.asarray(
                        result["test_store"]
                    ).tolist(),

                    "selected_features": selected_features
                }

                if model_name == "LSTM":

                    save_result["lookback"] = int(
                        lookback
                    )

                    save_result["epochs"] = int(
                        epochs
                    )

                    save_result["batch_size"] = int(
                        batch_size
                    )

                elif model_name == "Random Forest":

                    save_result["n_estimators"] = int(
                        n_estimators
                    )

                    save_result["max_depth"] = int(
                        max_depth
                    )

                    save_result["min_samples_split"] = int(
                        min_samples_split
                    )

                    save_result["min_samples_leaf"] = int(
                        min_samples_leaf
                    )

                elif model_name == "XGBoost":

                    save_result["n_estimators"] = int(
                        xgb_n_estimators
                    )

                    save_result["max_depth"] = int(
                        xgb_max_depth
                    )

                    save_result["learning_rate"] = float(
                        xgb_learning_rate
                    )

                    save_result["subsample"] = float(
                        xgb_subsample
                    )

                json_data = json.dumps(
                    save_result,
                    indent=4,
                    ensure_ascii=False
                )

                st.download_button(
                    "💾 Sonucu JSON Olarak Kaydet",
                    data=json_data,
                    file_name=f"{model_name}_sonuc.json",
                    mime="application/json",
                    use_container_width=True
                )


# ========================================================
# SONUÇLARI GÖSTER
# JSON'DAN GELSE DE NORMAL EĞİTİMDEN GELSE DE ÇALIŞIR
# ========================================================

if "model_result" in st.session_state:

    result = st.session_state["model_result"]

    st.divider()

    st.markdown(
        "<h2 style='text-align:center;'>📊 Model Sonuçları</h2>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "MAPE",
            f"{result['mape']:.2f}%"
        )

    with col2:

        st.metric(
            "MAE",
            f"{result['mae']:,.2f}"
        )

    with col3:

        st.metric(
            "RMSE",
            f"{result['rmse']:,.2f}"
        )

    with col4:

        st.metric(
            "R²",
            f"{result['r2']:.4f}"
        )

    # ====================================================
    # GENEL GERÇEK VS TAHMİN
    # ====================================================

    st.divider()

    st.markdown(
        "<h3 style='text-align:center;'>📈 Genel - Gerçek vs Tahmin</h3>",
        unsafe_allow_html=True
    )

    general_chart = pd.DataFrame({

        "Gerçek Satış":
            result["y_test"].values,

        "Tahmin Edilen Satış":
            result["y_pred"]

    })

    st.line_chart(
        general_chart
    )

    # ====================================================
    # MAĞAZA BAZLI
    # ====================================================

    if (
        result.get("test_store") is not None
        and result["test_store"].notna().any()
    ):

        st.divider()

        st.markdown(
            "<h2 style='text-align:center;'>🏪 Mağaza Bazlı Model Performansı</h2>",
            unsafe_allow_html=True
        )

        available_stores = sorted(
            result["test_store"].dropna().unique()
        )

        if len(available_stores) > 0:

            selected_store = st.selectbox(
                "İncelenecek mağazayı seçin:",
                available_stores
            )

            store_mask = (
                result["test_store"] == selected_store
            )

            store_y_test = result["y_test"][store_mask]

            store_y_pred = pd.Series(
                result["y_pred"],
                index=result["test_store"].index
            )[store_mask]

            store_mae = mean_absolute_error(
                store_y_test,
                store_y_pred
            )

            store_rmse = np.sqrt(
                mean_squared_error(
                    store_y_test,
                    store_y_pred
                )
            )

            store_r2 = r2_score(
                store_y_test,
                store_y_pred
            )

            store_mape = np.mean(
                np.abs(
                    (store_y_pred - store_y_test)
                    / store_y_test
                )
            ) * 100

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "MAPE",
                    f"{store_mape:.2f}%"
                )

            with col2:
                st.metric(
                    "MAE",
                    f"{store_mae:,.2f}"
                )

            with col3:
                st.metric(
                    "RMSE",
                    f"{store_rmse:,.2f}"
                )

            with col4:
                st.metric(
                    "R²",
                    f"{store_r2:.4f}"
                )

            st.divider()

            st.markdown(
                f"<h3 style='text-align:center;'>📈 Mağaza {selected_store} - Gerçek vs Tahmin</h3>",
                unsafe_allow_html=True
            )

            store_chart = pd.DataFrame({

                "Gerçek Satış":
                    store_y_test.values,

                "Tahmin Edilen Satış":
                    store_y_pred.values

            })

            st.line_chart(
                store_chart
            )
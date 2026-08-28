import streamlit as st
import pandas as pd
from prediction import run_prediction

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>📊 Sales Forecasting</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Mağaza Satışları ve Tahmin Analiz Paneli</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "CSV dosyanızı yükleyin",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("CSV başarıyla yüklendi!")


    # VERİ SETİ

    with st.expander("📋 Veri Seti Bilgileri", expanded=True):

        st.markdown(
            "<h2 style='text-align: center;'>Veri Seti</h2>",
            unsafe_allow_html=True
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


        # VERİ SETİ BİLGİLERİ

        st.divider()

        st.markdown(
            "<h3 style='text-align: center;'>📋 Veri Seti Bilgileri</h3>",
            unsafe_allow_html=True
        )

        row_count = df.shape[0]
        column_count = df.shape[1]

        store_count = df["Store"].nunique()

        min_date = pd.to_datetime(
            df["Date"]
        ).min()

        max_date = pd.to_datetime(
            df["Date"]
        ).max()

        missing_count = df.isnull().sum().sum()


        col1, col2, col3, col4, col5 = st.columns(5)


        with col1:
            st.metric(
                "Satır Sayısı",
                f"{row_count:,}"
            )


        with col2:
            st.metric(
                "Sütun Sayısı",
                column_count
            )


        with col3:
            st.metric(
                "Mağaza Sayısı",
                store_count
            )


        with col4:
            st.metric(
                "Başlangıç Tarihi",
                min_date.strftime("%d-%m-%Y")
            )


        with col5:
            st.metric(
                "Bitiş Tarihi",
                max_date.strftime("%d-%m-%Y")
            )

        # EKSİK VERİ

        if missing_count == 0:

            st.success(
                "✅ Veri setinde eksik veri bulunmamaktadır."
            )

        else:

            st.warning(
                f"⚠️ Veri setinde toplam {missing_count} eksik değer bulunmaktadır."
            )


        # EKSİK VERİ DETAYI

        missing_data = df.isnull().sum()

        missing_data = missing_data[
            missing_data > 0
        ]

        if not missing_data.empty:

            st.subheader(
                "⚠️ Eksik Veri Detayı"
            )

            st.dataframe(
                missing_data.rename(
                    "Eksik Değer Sayısı"
                ),
                use_container_width=True
            )

    # GENEL SATIŞ ANALİZİ
    
    with st.expander("📊 Genel Satış Analizi", expanded=True):
            
        st.markdown(
            "<h2 style='text-align: center;'>📊 Genel Satış Analizi</h2>",
            unsafe_allow_html=True
        )

        #Satış istatistikleri

        total_sales = df["Weekly_Sales"].sum()
        average_sales = df["Weekly_Sales"].mean()
        max_sales = df["Weekly_Sales"].max()
        min_sales = df["Weekly_Sales"].min()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💰 Toplam Satış",
                f"{total_sales:,.0f}"
            )

        with col2:
            st.metric(
                "📊 Ortalama Haftalık Satış",
                f"{average_sales:,.0f}"
            )

        with col3:
            st.metric(
                "⬆️ En Yüksek Satış",
                f"{max_sales:,.0f}"
            )

        with col4:
            st.metric(
                "⬇️ En Düşük Satış",
                f"{min_sales:,.0f}"
            )

        # MAĞAZA BAZINDA SATIŞ KARŞILAŞTIRMASI

        st.divider()

        st.markdown(
            "<h3 style='text-align: center;'>🏪 Mağaza Bazında Satışlar</h3>",
            unsafe_allow_html=True
        )
        
        store_sales = (
            df.groupby("Store")["Weekly_Sales"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            store_sales
        )

        # HAFTALIK SATIŞ GRAFİĞİ

        st.divider()

        st.markdown(
            "<h3 style='text-align: center;'>📈 Haftalık Satış Trendi</h3>",
            unsafe_allow_html=True
        )

        weekly_sales = (
            df.groupby("Date")["Weekly_Sales"]
            .sum()
            .sort_index()
        )

        st.line_chart(
            weekly_sales
        )

        # AYLIK SATIŞ GRAFİĞİ

        st.divider()

        st.markdown(
            "<h3 style='text-align: center;'>📅 Aylık Satış Trendi</h3>",
            unsafe_allow_html=True
        )

        df["Date"] = pd.to_datetime(df["Date"])

        monthly_sales = (
            df.groupby(df["Date"].dt.to_period("M"))["Weekly_Sales"]
            .sum()
        )

        monthly_sales.index = (
            monthly_sales.index.astype(str)
        )

        st.line_chart(
            monthly_sales
        )

    # MAĞAZA DETAYLI ANALİZ

    with st.expander("🏪 Mağaza Detaylı Analiz", expanded=True):

        st.markdown(
            "<h1 style='text-align: center;'>🏪 Mağaza Detaylı Analiz</h1>",
            unsafe_allow_html=True
        )
        selected_store = st.selectbox(
            "İncelemek istediğiniz mağazayı seçin:",
            sorted(df["Store"].unique())
        )

        store_df = df[
            df["Store"] == selected_store
        ].copy()

        # SEÇİLEN MAĞAZANIN İSTATİSTİKLERİ

        store_total_sales = (
            store_df["Weekly_Sales"].sum()
        )

        store_average_sales = (
            store_df["Weekly_Sales"].mean()
        )

        store_max_sales = (
            store_df["Weekly_Sales"].max()
        )

        store_min_sales = (
            store_df["Weekly_Sales"].min()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💰 Toplam Satış",
                f"{store_total_sales:,.0f}"
            )

        with col2:
            st.metric(
                "📊 Ortalama Satış",
                f"{store_average_sales:,.0f}"
            )

        with col3:
            st.metric(
                "⬆️ En Yüksek Satış",
                f"{store_max_sales:,.0f}"
            )

        with col4:
            st.metric(
                "⬇️ En Düşük Satış",
                f"{store_min_sales:,.0f}"
            )

        # SEÇİLEN MAĞAZANIN HAFTALIK SATIŞ TRENDİ

        st.divider()

        st.markdown(
            f"<h3 style='text-align:center;'>📈 Mağaza {selected_store} Haftalık Satış Trendi</h3>",
            unsafe_allow_html=True
        )
        
        store_df = store_df.sort_values("Date")

        store_sales_trend = (
            store_df
            .set_index("Date")["Weekly_Sales"]
        )

        st.line_chart(
            store_sales_trend
        )

        # SEÇİLEN MAĞAZANIN AYLIK SATIŞ TRENDİ

        st.divider()

        st.markdown(
            f"<h3 style='text-align: center;'>📅 Mağaza {selected_store} Aylık Satış Trendi</h3>",
            unsafe_allow_html=True
        )
        
        store_monthly_sales =(
            store_df
            .groupby(store_df["Date"].dt.to_period("M"))["Weekly_Sales"]
            .sum()
        )

        store_monthly_sales.index = (
            store_monthly_sales.index.astype(str)
        )

        st.line_chart(
            store_monthly_sales
        )

        # MAĞAZA SIRALAMASI

        store_ranking = (
            df.groupby("Store")["Weekly_Sales"]
            .mean()
            .sort_values(ascending=False)
        )

        store_rank = (
            store_ranking
            .rank(
                method="min",
                ascending=False
            )[selected_store]
        )

        st.info(
            f"🏆 Mağaza {selected_store}, "
            f"ortalama haftalık satış açısından "
            f"toplam {len(store_ranking)} mağaza içerisinde "
            f"{int(store_rank)}. sırada "
        )

    # TAHMİN
    
    with st.expander("🔮 Satış Tahmini", expanded=True):
        st.markdown(
            "<h3 style='text-align:center;'>🔮 Satış Tahmini</h3>",
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)

        with col1:

            forecast_store = st.selectbox(
                "Tahmin yapılacak mağaza:",
                sorted(df["Store"].unique()),
                key="forecast_store"
            )

        with col2:

            forecast_horizon = st.selectbox(
                "Kaç hafta ileriye tahmin edilsin?",
                [1, 2, 3, 4, 5, 6, 7, 8],
                index=3
            )

        st.write("")

        predict_button = st.button(
            "🔮 Tahmin Et",
            type="primary"
        )

        # TAHMİN SONUÇLARI

        if predict_button:

            with st.spinner("Tahmin yapılıyor..."):

                result = run_prediction(
                    df,
                    forecast_store,
                    forecast_horizon
                )

            st.success("Tahmin başarıyla tamamlandı!")

            # MODEL METRİKLERİ

            st.divider()

            st.markdown(
                "<h3 style='text-align: center;'>📊 Model Performansı</h3>",
                unsafe_allow_html=True
            )
            
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "MAPE",
                    f"{result['mape']:,.2f}"
                )

            with col2:
                st.metric(
                    "MAE",
                    f"{result['mae']:,.0f}"
                )

            with col3:
                st.metric(
                    "RMSE",
                    f"{result['rmse']:,.0f}"
                )

            with col4:
                st.metric(
                    "R²",
                    f"{result['r2']:.4f}"
                )


            # GERÇEK VS TAHMİN

            st.divider()

            st.markdown(
                "<h3 style='text-align: center;'>📈 Gerçek Satış vs Tahmin Edilen Satış</h3>",
                unsafe_allow_html=True
            )

            results = result["results"].copy()

            results["Date"] = pd.to_datetime(
                results["Date"]
            )

            results = results.set_index(
                "Date"
            )

            st.line_chart(
                results[
                    [
                        "Actual_Sales",
                        "Predicted_Sales"
                    ]
                ]
            )

            # TAHMİN TABLOSU

            st.divider()

            st.markdown(
                "<h3 style='text-align:center;'>📋 Tahmin Sonuçları</h3>",
                unsafe_allow_html=True
            )

            display_results = result["results"].copy()

            display_results["Actual_Sales"] = (
                display_results["Actual_Sales"]
                .round(0)
            )

            display_results["Predicted_Sales"] = (
                display_results["Predicted_Sales"]
                .round(0)
            )

            st.dataframe(
                display_results,
                use_container_width=True
            )

            # ========================================================
            # TÜM MAĞAZALARIN ORTALAMA MODEL PERFORMANSI
            # ========================================================

            st.divider()

            st.markdown(
                "<h3 style='text-align: center;'>📊 Tüm Mağazaların Ortalama Model Performansı</h3>",
                unsafe_allow_html=True
            )

            average_metrics = result["average_metrics"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Ortalama MAPE",
                    f"{average_metrics['MAPE']:.2f}%"
                )

            with col2:
                st.metric(
                    "Ortalama MAE",
                    f"{average_metrics['MAE']:,.0f}"
                )

            with col3:
                st.metric(
                    "Ortalama RMSE",
                    f"{average_metrics['RMSE']:,.0f}"
                )

            with col4:
                st.metric(
                    "Ortalama R²",
                    f"{average_metrics['R2']:.4f}"
                )
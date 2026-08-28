# Financial Sales Forecasting and Machine Learning Platform (My internship project)

Developed a machine learning-based financial sales forecasting application using Python and Streamlit. The project provides an interactive environment where users can upload their own training and test CSV datasets, define the target variable, select features manually or using the mRMR feature selection method, and train different regression models.

The application supports **Linear Regression, Random Forest, XGBoost, and LSTM** models. Model-specific hyperparameters can be configured through the user interface, and model performance is evaluated using **MAPE, MAE, RMSE, and R²** metrics.

For time-series forecasting, an **LSTM-based architecture** was implemented using a configurable lookback window. Historical sales values are incorporated into the input sequences, while numerical features are standardized using separate feature and target scalers. The LSTM can handle both datasets containing independent stores/branches and datasets without store information. When store information is available, the user can specify the corresponding column name, allowing the model to process each store as an independent time series.

The application also includes automatic date-based feature engineering, including **Year, Month, Week, Week Sin/Cos, Month Start, and Month End** features. Users can specify their own date column instead of relying on a fixed column name, making the application more flexible for different datasets.

Model results can be visualized through actual-vs-predicted sales charts and store-level performance analysis. Trained model results and their evaluation metrics can also be exported as JSON and loaded back into the application for later analysis.

The main goal of the project is to create a **flexible and user-friendly financial forecasting platform** that can work with different datasets while providing multiple machine learning approaches for comparison.


app.py and modeling_app.py are different applications that I made.
Predicting Disease Outbreaks Using Google Trends Data
Project Overview
This project aims to predict disease outbreaks using historical data from Google Trends. By leveraging time-series forecasting techniques, the model can provide early warnings of potential disease outbreaks by analyzing search patterns for relevant disease-related terms. The project utilizes various models, including Prophet, ARIMA, and hybrid models, to forecast future trends.

Key Features
Google Trends Data: Collected data using the pytrends library to analyze search trends over time.
Time Series Forecasting: Implemented forecasting models such as ARIMA, Prophet, and hybrid approaches to predict future disease outbreaks.
Model Evaluation: Compared different models based on accuracy, RMSE, and other performance metrics.
Streamlit Integration: Deployed the model on a Streamlit web app for easy interaction and visualization.
Libraries Used
pytrends: For retrieving Google Trends data.
prophet: A time-series forecasting model developed by Facebook.
statsmodels: To implement statistical models like ARIMA.
pmdarima: For automating ARIMA model selection.
joblib: For saving and loading machine learning models.
streamlit: To deploy the model and interact with it via a web interface.
Installation
To install the required dependencies, run:

bash
Copy code
pip install pytrends prophet statsmodels joblib pmdarima streamlit
Usage
Data Collection:

The pytrends library is used to collect historical search data from Google Trends.
Model Training:

Train the time-series models (Prophet, ARIMA) using the historical data.
Fine-tune model hyperparameters for better accuracy.
Prediction:

Use the trained models to forecast future disease outbreak trends.
Evaluate the model using metrics such as RMSE and MAE.
Visualization:

Visualize the forecasting results using Matplotlib or deploy the model via Streamlit to create an interactive web app.
Example
python
Copy code
from pytrends.request import TrendReq
from fbprophet import Prophet

# Sample Google Trends query
pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ['flu symptoms']
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='US', gprop='')

# Retrieve the data
data = pytrends.interest_over_time()

# Prophet model for forecasting
model = Prophet()
model.fit(data)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Visualize results
model.plot(forecast)
Results
The results show accurate disease trend forecasts, with the hybrid models (combining ARIMA and Prophet) showing the best performance for predicting disease outbreaks. The web app developed using Streamlit allows users to interact with the model and visualize future predictions.

Future Work
Model Refinement: Further improve model accuracy by incorporating more granular data and advanced hybrid modeling techniques.
Real-Time Monitoring: Set up a system for real-time data collection and forecasting to provide timely alerts on potential outbreaks.
Expansion: Apply the model to other regions or diseases for broader applicability.
Contributors
Abisek Raut : Data Science, Modeling, and Web App Development

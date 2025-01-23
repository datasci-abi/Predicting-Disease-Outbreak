# Predicting Disease Outbreaks Using Google Trends Data

## Project Overview
This project leverages historical Google Trends data to forecast potential disease outbreaks. By analyzing search patterns for disease-related terms, the model employs time-series forecasting techniques to provide early warnings, enhancing public health responsiveness.

### Key Features
- **Google Trends Data**: Uses the `pytrends` library to analyze search trends over time.
- **Time-Series Forecasting**: Implements models such as ARIMA, Prophet, and hybrid approaches to predict future disease outbreaks.
- **Model Evaluation**: Compares models based on accuracy metrics like RMSE and MAE.
- **Streamlit Integration**: Deploys the model in a Streamlit web app for interactive visualization and easy user interaction.

## Libraries Used
- `pytrends`: For retrieving Google Trends data.
- `prophet`: Developed by Facebook for time-series forecasting.
- `statsmodels`: Implements statistical models, including ARIMA.
- `pmdarima`: Automates ARIMA model selection.
- `joblib`: Manages saving and loading of models.
- `streamlit`: Facilitates model deployment and web interaction.

## Installation
To install the required dependencies, run the following command:
```bash
pip install pytrends prophet statsmodels pmdarima joblib streamlit
Usage
Data Collection:
python

from pytrends.request import TrendReq

# Setup and send a query to Google Trends
pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ['flu symptoms']
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='US', gprop='')
data = pytrends.interest_over_time()
Model Training:
Train the time-series models (Prophet, ARIMA) using the historical data and fine-tune hyperparameters for better accuracy.

Prediction:
python

from fbprophet import Prophet

# Prophet model for forecasting
model = Prophet()
model.fit(data)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Visualize the results
model.plot(forecast)
Visualization:
Deploy the model via Streamlit to create an interactive web app, or use Matplotlib for static visualization.

Results
The models, particularly the hybrid models combining ARIMA and Prophet, show strong performance in predicting disease outbreaks. The Streamlit web app allows for dynamic interaction and visualization of predicted trends.

Future Work
Model Refinement: Enhance model accuracy with more granular data and advanced modeling techniques.
Real-Time Monitoring: Develop a system for real-time data collection and forecasting.
Expansion: Extend the model to additional regions or diseases to increase its applicability.
Contributors
Abisek Raut: Data Science, Modeling, and Web App Development
vbnet



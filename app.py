from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import io, base64
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import datetime
import random
'''
# Map common names to tickers (you can expand this or use APIs later)
company_lookup = {
    "apple": "AAPL",
    "tcs": "TCS.NS",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "reliance": "RELIANCE.NS",
    # Add more...
}
def extract_tickers_from_text(text):
    found = []
    text_lower = text.lower()
    for name in company_lookup:
        if name in text_lower:
            found.append((name, company_lookup[name]))
    return found
def get_stock_summary(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("shortName"),
            "ticker": ticker,
            "currentPrice": info.get("currentPrice"),
            "currency": info.get("currency"),
            "summary": info.get("longBusinessSummary"),
            "previousClose": info.get("previousClose"),
            "priceChange": round(info.get("currentPrice", 0) - info.get("previousClose", 0), 2)
        }
    except Exception as e:
        return {"error": str(e)}'''




app = Flask(__name__)

# Load models
lstm_model = load_model('models/stock_prediction_lstm.h5')
bilstm_model = load_model('models/stock_prediction_Bilstm.h5')

def get_model(name):
    if name == 'LSTM':
        return lstm_model
    elif name == 'BiLSTM':
        return bilstm_model
    return None

def get_prediction_plot(actual, predicted_price):
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64

    actual = np.array(actual).flatten()

    plt.figure(figsize=(8, 4))
    plt.plot(actual, label="Actual", color="blue")

    # Plot the predicted point
    if len(actual) > 0:
        last_index = len(actual) - 1
        plt.plot(last_index + 1, predicted_price, 'o', color='orange', label='Predicted')
        plt.plot([last_index, last_index + 1], [actual[-1], predicted_price], linestyle='--', color='orange')

    plt.title("Actual vs Predicted Stock Price")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64





def get_history_plot(df):
    plt.figure(figsize=(6, 3))
    plt.plot(df['Close'], label='Close Price')
    plt.title("Historical Close Price")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64




@app.route('/', methods=['GET', 'POST'])
def index():
    ticker = None
    predicted_price = None
    error = None
    graph_img = None
    history_img = None
    stock_summaries = []

    # Company name to ticker mapping
    company_lookup = {
        # US Stocks
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "facebook": "META",
    "twitter": "TWTR",
    "netflix": "NFLX",
    "walmart": "WMT",
    "intel": "INTC",
    "coca-cola": "KO",
    "nike": "NKE",
    "disney": "DIS",
    "starbucks": "SBUX",
    "pepsi": "PEP",
    "boeing": "BA",
    "nvidia": "NVDA",
    "spotify": "SPOT",
    "salesforce": "CRM",
    
    # Indian Stocks (NSE)
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "reliance": "RELIANCE.NS",
    "hdfc": "HDFC.NS",
    "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "axis bank": "AXISBANK.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "sun pharma": "SUNPHARMA.NS",
    "l&t": "LT.NS",
    "maruti": "MARUTI.NS",
    "hindalco": "HINDALCO.NS",
    "kotak mahindra": "KOTAKBANK.NS",
    "nestle india": "NESTLEIND.NS",
    "tata motors": "TATAMOTORS.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "mahilakshmi": "MAHINDCIE.NS",
    "shree cement": "SHREECEM.NS",
    
    # UK Stocks
    "vodafone": "VOD.L",
    "tesco": "TSCO.L",
    "barclays": "BARC.L",
    "unilever": "ULVR.L",
    "bp": "BP.L",
    "glaxo smithkline": "GSK.L",
    
    # European Stocks
    "nestle": "NESN.SW",  # Switzerland
    "siemens": "SIE.DE",  # Germany
    "l'oréal": "OR.PA",  # France
    "sanofi": "SAN.PA",  # France
    "sap": "SAP.DE",  # Germany
    "asml": "ASML.AS",  # Netherlands
    
    # Chinese Stocks
    "alibaba": "BABA",
    "tencent": "0700.HK",
    "baidu": "BIDU",
    "jd.com": "JD",
    "meituan": "3690.HK",
    "pinduoduo": "PDD",
    
    # Other global stocks
    "sony": "6758.T",  # Japan
    "toshiba": "6502.T",  # Japan
    "canon": "7751.T",  # Japan
    "lg electronics": "066570.KS",  # South Korea
    "samsung electronics": "005930.KS",  # South Korea
    "bayer": "BAYN.DE",  # Germany
    "abbvie": "ABBV",  # USA
    "ford": "F",  # USA
    "general electric": "GE",  # USA
    "abbott laboratories": "ABT",  # USA
    "general motors": "GM",  # USA
    "3m": "MMM"  # USA
    }

    def extract_tickers_from_text(text):
        found = []
        text_lower = text.lower()
        for name in company_lookup:
            if name in text_lower:
                found.append((name, company_lookup[name]))
        return found

    def get_stock_summary(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            name = info.get("shortName", ticker)
            current_price = info.get("currentPrice")
            previous_close = info.get("previousClose")
            open_price = info.get("open")
            day_low = info.get("dayLow")
            day_high = info.get("dayHigh")
            fifty_two_week_low = info.get("fiftyTwoWeekLow")
            fifty_two_week_high = info.get("fiftyTwoWeekHigh")
            volume = info.get("volume")
            avg_volume = info.get("averageVolume")
            market_cap = info.get("marketCap")
            currency = info.get("currency", "USD")

            change = round(current_price - previous_close, 2)
            change_percent = round((change / previous_close) * 100, 2) if previous_close else 0

            if change > 0:
                trend = f"{name} is currently seeing a positive uptick, trading at {current_price} {currency}, up {change_percent}% from the previous close."
            elif change < 0:
                trend = f"{name} is facing a bit of a dip today, down {abs(change_percent)}% at {current_price} {currency}."
            else:
                trend = f"{name} is holding steady today at {current_price} {currency}, showing no major price movement."

            volatility = "showing steady movement" if abs(day_high - day_low) < 0.5 else "experiencing some volatility"
            volume_comment = (
                "with trading volume close to the daily average."
                if volume and avg_volume and abs(volume - avg_volume) / avg_volume < 0.2
                else "with an unusual spike in trading activity."
            )

            commentary = (
                f"{trend} The stock opened at {open_price}, with a day range between {day_low} and {day_high}, "
                f"{volatility}. It’s currently {volume_comment} Over the past year, the stock has ranged from {fifty_two_week_low} to {fifty_two_week_high}."
            )

            return {
                "name": name,
                "ticker": ticker,
                "currentPrice": current_price,
                "currency": currency,
                "summary": commentary,
                "priceChange": f"{change:+} ({change_percent:+}%)",
            }

        except Exception as e:
            return {"error": str(e)}

    if request.method == 'POST':
        ticker_input = request.form.get('ticker_input', '').strip()
        question_input = request.form.get('question_input', '').strip()
        model_name = request.form.get('model', 'LSTM')

        if question_input:
            companies = extract_tickers_from_text(question_input)
            for name, tick in companies:
                summary = get_stock_summary(tick)
                if "error" not in summary:
                    stock_summaries.append(summary)
            return render_template('index.html', predicted_price=None, error=None,
                                   graph_img=None, history_img=None,
                                   ticker=None, stock_summaries=stock_summaries)

        elif ticker_input:
            ticker = ticker_input.upper()
            model = get_model(model_name)

            try:
                df = yf.download(ticker, start="2012-01-01", end=datetime.date.today())
                if df.empty or 'Close' not in df.columns:
                    raise ValueError("No data found")

                data = df[['Close']].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)

                input_seq = scaled_data[-100:]
                X_input = np.reshape(input_seq, (1, 100, 1))
                prediction = model.predict(X_input)
                predicted_price = scaler.inverse_transform(prediction)[0][0]

                graph_img = get_prediction_plot(data[-60:], predicted_price)
                history_img = get_history_plot(df)

            except Exception as e:
                error = f"Error during prediction: {str(e)}"

    return render_template('index.html', predicted_price=predicted_price,
                           error=error, graph_img=graph_img, history_img=history_img,
                           ticker=ticker, stock_summaries=stock_summaries)


if __name__ == '__main__':
    app.run(debug=True)

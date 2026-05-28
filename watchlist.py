import os
import requests
import akshare as ak
import google.generativeai as genai

# ------------------
# Load environment secrets
# ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STOCK_LIST = os.getenv("STOCK_LIST")

stock_list = [x.strip() for x in STOCK_LIST.split(",")]

# ------------------
# Gemini AI setup
# ------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# ------------------
# Get stock data
# ------------------
def get_stock_info(symbol):
    try:
        df = ak.stock_hk_spot(symbol=symbol.replace(".HK", ""))
        name = df.iloc[0]["股票名称"]
        price = df.iloc[0]["最新价"]
        change = df.iloc[0]["涨跌幅"]
        return {"code": symbol, "name": name, "price": price, "change": change}
    except:
        return {"code": symbol, "name": "N/A", "price": "N/A", "change": "N/A"}

# ------------------
# AI analysis
# ------------------
def ai_analysis(stock):
    prompt = f"""
    Analyze this Hong Kong stock:
    Code: {stock['code']}
    Name: {stock['name']}
    Price: {stock['price']}
    Change: {stock['change']}

    Give a short 1-line comment.
    """
    try:
        res = model.generate_content(prompt)
        return res.text.strip()
    except:
        return "AI analysis unavailable"

# ------------------
# Send Telegram
# ------------------
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=10)
    except:
        print("Telegram send failed")

# ------------------
# Main report
# ------------------
report = "📊 HK Stock Watchlist Report\n\n"

for code in stock_list:
    stock = get_stock_info(code)
    comment = ai_analysis(stock)
    report += f"🔹 {code} | {stock['name']}\nPrice: {stock['price']} | Change: {stock['change']}\nComment: {comment}\n\n"

report += "⚠️ AI for reference only, not investment advice."

print(report)
send_telegram(report)

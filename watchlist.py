import os
import requests
import yfinance as yf
import google.genai as genai

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
client = genai.Client(api_key=GEMINI_API_KEY)

# ------------------
# Get stock data with yfinance (稳定版本)
# ------------------
def get_stock_info(symbol):
    try:
        # 把 0001.HK 转换成 yfinance 支持的格式
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        name = info.get("longName", "N/A")
        price = info.get("regularMarketPrice", "N/A")
        prev_close = info.get("previousClose", "N/A")
        
        if price != "N/A" and prev_close != "N/A":
            change_pct = round((price - prev_close) / prev_close * 100, 2)
        else:
            change_pct = "N/A"
        
        return {"code": symbol, "name": name, "price": price, "change": change_pct}
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return {"code": symbol, "name": "N/A", "price": "N/A", "change": "N/A"}

# ------------------
# AI analysis
# ------------------
def ai_analysis(stock):
    if stock["price"] == "N/A":
        return "数据获取失败，无法分析"
    prompt = f"""
    分析这只港股：
    代码: {stock['code']}
    名称: {stock['name']}
    最新价: {stock['price']}
    涨跌幅: {stock['change']}%

    用一句话给出简短的市场解读。
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"AI分析失败: {e}")
        return "AI分析暂时不可用"

# ------------------
# Send Telegram
# ------------------
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"Telegram推送失败: {e}")

# ------------------
# Main report
# ------------------
report = "📊 港股观察名单报告\n\n"

for code in stock_list:
    stock = get_stock_info(code)
    comment = ai_analysis(stock)
    report += f"🔹 {code} | {stock['name']}\n价格: {stock['price']} | 涨跌幅: {stock['change']}%\n解读: {comment}\n\n"

report += "⚠️ AI分析仅供参考，不构成投资建议。"

print(report)
send_telegram(report)

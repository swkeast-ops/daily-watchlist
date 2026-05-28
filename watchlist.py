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
# Get ALL stock data FIRST (only 1 API call per stock)
# ------------------
def get_all_stock_data():
    all_data = []
    for symbol in stock_list:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            name = info.get("longName", symbol)
            price = info.get("regularMarketPrice", "N/A")
            prev_close = info.get("previousClose", "N/A")
            
            if price != "N/A" and prev_close != "N/A" and prev_close != 0:
                change_pct = round((price - prev_close) / prev_close * 100, 2)
            else:
                change_pct = "N/A"
            
            all_data.append({
                "code": symbol,
                "name": name,
                "price": price,
                "change": change_pct
            })
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            all_data.append({
                "code": symbol,
                "name": symbol,
                "price": "N/A",
                "change": "N/A"
            })
    return all_data

# ------------------
# AI analysis: ONLY 1 API CALL FOR ALL STOCKS
# ------------------
def generate_full_report(all_stocks):
    # 把所有股票数据整理成表格字符串
    stock_table = "代码 | 名称 | 最新价 | 涨跌幅%\n"
    stock_table += "---|---|---|---\n"
    
    for stock in all_stocks:
        stock_table += f"{stock['code']} | {stock['name']} | {stock['price']} | {stock['change']}\n"
    
    prompt = f"""
    你是一个专业的港股分析师。请分析以下港股观察名单，生成一份简洁清晰的每日报告：

    {stock_table}

    要求：
    1.  首先列出**涨幅前5名**和**跌幅前5名**的股票
    2.  对每只涨跌幅度较大的股票，用一句话给出简短的市场解读
    3.  最后给出一个整体市场小结
    4.  语言简洁，重点突出，适合在手机上阅读
    5.  不要使用Markdown表格，用纯文本格式
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"AI分析失败: {e}")
        # 如果AI失败，返回原始数据列表
        fallback_report = "📊 港股观察名单报告（AI分析暂时不可用）\n\n"
        for stock in all_stocks:
            fallback_report += f"🔹 {stock['code']} | {stock['name']}\n价格: {stock['price']} | 涨跌幅: {stock['change']}%\n\n"
        return fallback_report

# ------------------
# Send Telegram
# ------------------
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # 长消息用HTML格式，避免Markdown解析错误
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"Telegram推送失败: {e}")

# ------------------
# Main
# ------------------
if __name__ == "__main__":
    print("📥 开始获取股票数据...")
    all_stocks = get_all_stock_data()
    
    print("🤖 开始AI分析（仅1次API调用）...")
    report = generate_full_report(all_stocks)
    
    print("📤 发送报告到Telegram...")
    send_telegram(report)
    
    print("✅ 运行完成！")
    print("\n报告内容：")
    print(report)

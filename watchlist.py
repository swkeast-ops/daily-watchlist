import os
import requests
import time
import yfinance as yf
import google.genai as genai
from google.genai import types

# ------------------
# Load environment secrets
# ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STOCK_LIST = os.getenv("STOCK_LIST")

stock_list = [x.strip() for x in STOCK_LIST.split(",")]

# ------------------
# Gemini AI setup with auto-retry and fallback
# ------------------
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_content_with_retry(prompt, max_retries=3):
    """带自动重试和模型降级的AI调用函数"""
    models = ["gemini-2.5-flash", "gemini-1.5-flash"]
    retry_delay = 2  # 初始等待2秒
    
    for model in models:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        top_p=0.1,
                        top_k=1
                    )
                )
                print(f"✅ AI分析成功，使用模型: {model}")
                return response.text.strip()
            except Exception as e:
                print(f"❌ 模型 {model} 第 {attempt+1} 次调用失败: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
    
    print("❌ 所有模型调用都失败了")
    return None

# ------------------
# Get ALL stock data (8个核心指标)
# ------------------
def get_all_stock_data():
    all_data = []
    for symbol in stock_list:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 基础价格数据
            price = info.get("regularMarketPrice", "N/A")
            prev_close = info.get("previousClose", "N/A")
            change_pct = round((price - prev_close) / prev_close * 100, 2) if price != "N/A" and prev_close != "N/A" else "N/A"
            
            # 成交量和换手率
            volume = info.get("regularMarketVolume", "N/A")
            shares_outstanding = info.get("sharesOutstanding", "N/A")
            turnover = round(volume / shares_outstanding * 100, 2) if volume != "N/A" and shares_outstanding != "N/A" else "N/A"
            
            # 52周高低点
            fifty_two_week_high = info.get("fiftyTwoWeekHigh", "N/A")
            fifty_two_week_low = info.get("fiftyTwoWeekLow", "N/A")
            
            # 20日均线（短期趋势）
            fifty_day_average = info.get("fiftyDayAverage", "N/A")
            trend = "上涨" if price > fifty_day_average else "下跌" if price < fifty_day_average else "横盘"
            
            all_data.append({
                "code": symbol,
                "name": info.get("longName", symbol),
                "price": price,
                "change_pct": change_pct,
                "volume": volume,
                "turnover": turnover,
                "fifty_two_week_high": fifty_two_week_high,
                "fifty_two_week_low": fifty_two_week_low,
                "trend": trend
            })
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            all_data.append({
                "code": symbol,
                "name": symbol,
                "price": "N/A",
                "change_pct": "N/A",
                "volume": "N/A",
                "turnover": "N/A",
                "fifty_two_week_high": "N/A",
                "fifty_two_week_low": "N/A",
                "trend": "N/A"
            })
    return all_data

# ------------------
# 终极防幻觉AI分析
# ------------------
def generate_full_report(all_stocks):
    # 筛选出有有效数据的股票
    valid_stocks = [s for s in all_stocks if s["price"] != "N/A"]
    
    # 按涨跌幅排序
    sorted_stocks = sorted(valid_stocks, key=lambda x: x["change_pct"], reverse=True)
    top_gainers = sorted_stocks[:5]
    top_losers = sorted_stocks[-5:]
    
    # 准备数据表格
    data_table = "代码 | 名称 | 最新价 | 涨跌幅% | 换手率% | 20日均线趋势 | 52周高低\n"
    data_table += "---|---|---|---|---|---|---\n"
    
    for stock in valid_stocks:
        high_low = f"{stock['fifty_two_week_low']}-{stock['fifty_two_week_high']}"
        data_table += f"{stock['code']} | {stock['name']} | {stock['price']} | {stock['change_pct']} | {stock['turnover']} | {stock['trend']} | {high_low}\n"
    
    prompt = f"""
    你是一个严格遵守事实的港股分析师。请基于我提供的以下数据，生成一份每日市场报告。

    【绝对规则 - 违反任何一条都视为失败】
    1.  所有分析必须100%基于我提供的数据，不得使用任何你自己的知识库
    2.  不得编造任何我没有提供的数据，包括新闻、事件、财务数据等
    3.  每一个结论都必须明确引用对应的指标
    4.  如果数据不足或不确定，明确标注"数据不足，无法分析"
    5.  不得给出任何投资建议，只做客观描述和分析
    6.  语言简洁，重点突出，适合在手机上阅读

    【股票数据】
    {data_table}

    【报告结构】
    1.  市场概览：简要说明今天整体市场的涨跌情况
    2.  涨幅前5名：列出涨幅最大的5只股票，结合换手率和趋势进行简要分析
    3.  跌幅前5名：列出跌幅最大的5只股票，结合换手率和趋势进行简要分析
    4.  特别关注：列出换手率超过5%的股票，说明可能有较大资金活动
    5.  免责声明：AI分析仅供参考，不构成投资建议

    请严格按照以上结构生成报告，不要添加任何额外内容。
    """
    
    ai_result = generate_content_with_retry(prompt)
    
    if ai_result:
        return ai_result
    else:
        # 终极降级：纯数据报告
        fallback_report = "📊 港股观察名单报告（AI分析暂时不可用）\n\n"
        fallback_report += "涨幅前5名：\n"
        for stock in top_gainers:
            fallback_report += f"🔹 {stock['code']} | {stock['name']} | {stock['change_pct']}%\n"
        fallback_report += "\n跌幅前5名：\n"
        for stock in top_losers:
            fallback_report += f"🔹 {stock['code']} | {stock['name']} | {stock['change_pct']}%\n"
        fallback_report += "\n⚠️ AI分析仅供参考，不构成投资建议。"
        return fallback_report

# ------------------
# Send Telegram
# ------------------
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
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
    
    print("🤖 开始AI分析（带自动重试和模型降级）...")
    report = generate_full_report(all_stocks)
    
    print("📤 发送报告到Telegram...")
    send_telegram(report)
    
    print("✅ 运行完成！")
    print("\n报告内容：")
    print(report)

import os
import requests
import time
import yfinance as yf
import pandas as pd
import numpy as np
import google.genai as genai
from google.genai import types
from datetime import datetime, timedelta

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

def generate_content_with_retry(prompt, max_retries=2):
    """带自动重试和模型降级的AI调用函数"""
    models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
    retry_delay = 3
    
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
                    retry_delay *= 2
    
    print("❌ 所有Gemini模型调用都失败了，切换到纯Python智能分析")
    return None

# ------------------
# 技术指标计算函数
# ------------------
def calculate_technical_indicators(hist):
    """计算所有技术指标"""
    indicators = {}
    
    try:
        # MACD
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        indicators['macd'] = round(macd.iloc[-1], 4)
        indicators['macd_signal'] = round(signal.iloc[-1], 4)
        indicators['macd_crossover'] = "金叉" if macd.iloc[-1] > signal.iloc[-1] else "死叉"
        
        # RSI(14)
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = round(rsi.iloc[-1], 2)
        
        # 布林带
        ma20 = hist['Close'].rolling(window=20).mean()
        std20 = hist['Close'].rolling(window=20).std()
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)
        current_price = hist['Close'].iloc[-1]
        bollinger_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        indicators['bollinger_position'] = round(bollinger_position * 100, 1)
        
        # 均线
        indicators['ma5'] = round(hist['Close'].rolling(window=5).mean().iloc[-1], 2)
        indicators['ma20'] = round(ma20.iloc[-1], 2)
        indicators['ma60'] = round(hist['Close'].rolling(window=60).mean().iloc[-1], 2)
        
        # 趋势判断
        if current_price > indicators['ma5'] > indicators['ma20'] > indicators['ma60']:
            indicators['trend'] = "强势上涨"
        elif current_price < indicators['ma5'] < indicators['ma20'] < indicators['ma60']:
            indicators['trend'] = "强势下跌"
        elif current_price > indicators['ma20']:
            indicators['trend'] = "短期上涨"
        elif current_price < indicators['ma20']:
            indicators['trend'] = "短期下跌"
        else:
            indicators['trend'] = "横盘震荡"
        
        # 成交量变化率
        current_volume = hist['Volume'].iloc[-1]
        avg_volume_20 = hist['Volume'].rolling(window=20).mean().iloc[-1]
        indicators['volume_change_pct'] = round((current_volume / avg_volume_20 - 1) * 100, 2)
        indicators['volume_ratio'] = round(current_volume / avg_volume_20, 2)
        
        # ATR(14) 波动率
        high_low = hist['High'] - hist['Low']
        high_close = np.abs(hist['High'] - hist['Close'].shift())
        low_close = np.abs(hist['Low'] - hist['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = round(true_range.rolling(window=14).mean().iloc[-1], 4)
        
    except Exception as e:
        print(f"技术指标计算失败: {e}")
        indicators = {
            'macd': "N/A", 'macd_signal': "N/A", 'macd_crossover': "N/A",
            'rsi': "N/A", 'bollinger_position': "N/A",
            'ma5': "N/A", 'ma20': "N/A", 'ma60': "N/A", 'trend': "N/A",
            'volume_change_pct': "N/A", 'volume_ratio': "N/A", 'atr': "N/A"
        }
    
    return indicators

# ------------------
# 获取美股隔夜数据
# ------------------
def get_us_market_data():
    """获取隔夜美股三大指数数据"""
    us_indices = {
        "^GSPC": "标普500",
        "^DJI": "道琼斯工业平均指数",
        "^IXIC": "纳斯达克综合指数"
    }
    
    us_data = []
    for symbol, name in us_indices.items():
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get("regularMarketPrice", "N/A")
            change_pct = round(info.get("regularMarketChangePercent", 0) * 100, 2)
            us_data.append({
                "name": name,
                "price": price,
                "change_pct": change_pct
            })
        except Exception as e:
            print(f"获取 {name} 数据失败: {e}")
            us_data.append({
                "name": name,
                "price": "N/A",
                "change_pct": "N/A"
            })
    
    return us_data

# ------------------
# Get ALL stock data
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
            
            # 基本面指标
            pe_ratio = info.get("trailingPE", "N/A")
            pb_ratio = info.get("priceToBook", "N/A")
            dividend_yield = round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else "N/A"
            roe = round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else "N/A"
            eps = info.get("trailingEps", "N/A")
            beta = info.get("beta", "N/A")
            
            # 成交量和换手率
            volume = info.get("regularMarketVolume", "N/A")
            shares_outstanding = info.get("sharesOutstanding", "N/A")
            turnover = round(volume / shares_outstanding * 100, 2) if volume != "N/A" and shares_outstanding != "N/A" else "N/A"
            
            # 52周高低点
            fifty_two_week_high = info.get("fiftyTwoWeekHigh", "N/A")
            fifty_two_week_low = info.get("fiftyTwoWeekLow", "N/A")
            if price != "N/A" and fifty_two_week_high != "N/A" and fifty_two_week_low != "N/A":
                fifty_two_week_position = round((price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low) * 100, 1)
            else:
                fifty_two_week_position = "N/A"
            
            # 获取历史K线数据计算技术指标
            hist = ticker.history(period="3mo")
            tech_indicators = calculate_technical_indicators(hist)
            
            # 合并所有数据
            stock_data = {
                "code": symbol,
                "name": info.get("longName", symbol),
                "price": price,
                "change_pct": change_pct,
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "dividend_yield": dividend_yield,
                "roe": roe,
                "eps": eps,
                "beta": beta,
                "turnover": turnover,
                "fifty_two_week_position": fifty_two_week_position,
                **tech_indicators
            }
            
            all_data.append(stock_data)
            
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            all_data.append({
                "code": symbol,
                "name": symbol,
                "price": "N/A",
                "change_pct": "N/A",
                "pe_ratio": "N/A",
                "pb_ratio": "N/A",
                "dividend_yield": "N/A",
                "roe": "N/A",
                "eps": "N/A",
                "beta": "N/A",
                "turnover": "N/A",
                "fifty_two_week_position": "N/A",
                'macd': "N/A", 'macd_signal': "N/A", 'macd_crossover': "N/A",
                'rsi': "N/A", 'bollinger_position': "N/A",
                'ma5': "N/A", 'ma20': "N/A", 'ma60': "N/A", 'trend': "N/A",
                'volume_change_pct': "N/A", 'volume_ratio': "N/A", 'atr': "N/A"
            })
    return all_data

# ------------------
# 纯Python智能分析引擎
# ------------------
def generate_python_analysis(all_stocks, is_morning, us_data=None):
    """完全不需要AI，用纯Python代码生成专业分析报告"""
    valid_stocks = [s for s in all_stocks if s["price"] != "N/A"]
    
    # 按涨跌幅排序
    sorted_stocks = sorted(valid_stocks, key=lambda x: x["change_pct"], reverse=True)
    top_gainers = sorted_stocks[:5]
    top_losers = sorted_stocks[-5:]
    
    # 计算市场整体情况
    avg_change = round(np.mean([s["change_pct"] for s in valid_stocks]), 2)
    up_count = len([s for s in valid_stocks if s["change_pct"] > 0])
    down_count = len([s for s in valid_stocks if s["change_pct"] < 0])
    
    # 筛选特别关注股票
    overbought = [s for s in valid_stocks if s["rsi"] != "N/A" and s["rsi"] > 70]
    oversold = [s for s in valid_stocks if s["rsi"] != "N/A" and s["rsi"] < 30]
    high_volume = [s for s in valid_stocks if s["volume_change_pct"] != "N/A" and s["volume_change_pct"] > 100]
    high_dividend = [s for s in valid_stocks if s["dividend_yield"] != "N/A" and s["dividend_yield"] > 5]
    
    # 获取当前日期
    today = datetime.now().strftime("%Y年%m月%d日")
    
    # 生成报告
    if is_morning:
        report = f"🌅 港股早盘观察报告\n\n"
        report += f"📅 日期: {today}\n\n"
        
        # 美股隔夜表现
        if us_data:
            report += "🇺🇸 隔夜美股表现\n"
            for index in us_data:
                report += f"{index['name']}: {index['change_pct']}%\n"
            report += "\n"
        
        report += "📈 昨日市场回顾\n"
        report += f"整体平均涨跌幅: {avg_change}%\n"
        report += f"上涨股票: {up_count} 只 | 下跌股票: {down_count} 只\n\n"
        
        report += "🚀 昨日涨幅前5名\n"
        for stock in top_gainers:
            report += f"🔹 {stock['code']} | {stock['name']} | {stock['change_pct']}%\n"
        
        report += "\n📉 昨日跌幅前5名\n"
        for stock in top_losers:
            report += f"🔹 {stock['code']} | {stock['name']} | {stock['change_pct']}%\n"
        
        report += "\n⚠️ 今日重点关注\n"
        if overbought:
            report += "RSI超买(>70): " + ", ".join([s["code"] for s in overbought]) + "\n"
        if oversold:
            report += "RSI超卖(<30): " + ", ".join([s["code"] for s in oversold]) + "\n"
        if high_volume:
            report += "昨日成交量翻倍: " + ", ".join([s["code"] for s in high_volume]) + "\n"
        
    else:
        report = f"🌇 港股收盘总结报告\n\n"
        report += f"📅 日期: {today}\n\n"
        
        report += "📈 今日市场概览\n"
        report += f"整体平均涨跌幅: {avg_change}%\n"
        report += f"上涨股票: {up_count} 只 | 下跌股票: {down_count} 只\n\n"
        
        report += "🚀 今日涨幅前5名\n"
        for stock in top_gainers:
            comment = ""
            if stock["volume_change_pct"] != "N/A" and stock["volume_change_pct"] > 50:
                comment += "成交量大幅放大"
            elif stock["trend"] == "强势上涨":
                comment += "处于强势上涨趋势"
            
            report += f"🔹 {stock['code']} | {stock['name']}\n"
            report += f"涨跌幅: {stock['change_pct']}% | 换手率: {stock['turnover']}%\n"
            if comment:
                report += f"分析: {comment}\n"
            report += "\n"
        
        report += "📉 今日跌幅前5名\n"
        for stock in top_losers:
            comment = ""
            if stock["volume_change_pct"] != "N/A" and stock["volume_change_pct"] > 50:
                comment += "放量下跌"
            elif stock["trend"] == "强势下跌":
                comment += "处于强势下跌趋势"
            
            report += f"🔹 {stock['code']} | {stock['name']}\n"
            report += f"涨跌幅: {stock['change_pct']}% | 换手率: {stock['turnover']}%\n"
            if comment:
                report += f"分析: {comment}\n"
            report += "\n"
        
        report += "⚠️ 特别关注\n"
        if overbought:
            report += "RSI超买(>70): " + ", ".join([s["code"] for s in overbought]) + "\n"
        if oversold:
            report += "RSI超卖(<30): " + ", ".join([s["code"] for s in oversold]) + "\n"
        if high_volume:
            report += "今日成交量翻倍: " + ", ".join([s["code"] for s in high_volume]) + "\n"
        if high_dividend:
            report += "高股息(>5%): " + ", ".join([s["code"] for s in high_dividend]) + "\n"
    
    report += "\n⚠️ 分析仅供参考，不构成投资建议。"
    
    return report

# ------------------
# 终极防幻觉AI分析
# ------------------
def generate_full_report(all_stocks, is_morning, us_data=None):
    valid_stocks = [s for s in all_stocks if s["price"] != "N/A"]
    
    # 按涨跌幅排序
    sorted_stocks = sorted(valid_stocks, key=lambda x: x["change_pct"], reverse=True)
    top_gainers = sorted_stocks[:5]
    top_losers = sorted_stocks[-5:]
    
    # 获取当前日期
    today = datetime.now().strftime("%Y年%m月%d日")
    
    # 准备数据表格
    data_table = "代码 | 名称 | 涨跌幅% | PE | PB | 股息率% | ROE% | RSI | 趋势 | 成交量变化%\n"
    data_table += "---|---|---|---|---|---|---|---|---|---\n"
    
    for stock in valid_stocks:
        data_table += f"{stock['code']} | {stock['name']} | {stock['change_pct']} | {stock['pe_ratio']} | {stock['pb_ratio']} | {stock['dividend_yield']} | {stock['roe']} | {stock['rsi']} | {stock['trend']} | {stock['volume_change_pct']}\n"
    
    # 构建不同的提示词
    if is_morning:
        us_market_summary = ""
        if us_data:
            us_market_summary = "隔夜美股表现：\n"
            for index in us_data:
                us_market_summary += f"{index['name']}: {index['change_pct']}%\n"
        
        prompt = f"""
        你是一个严格遵守事实的港股分析师。现在是香港时间早上9点，港股还没有开盘。请基于我提供的前一个交易日的收盘数据和隔夜美股数据，生成一份早盘观察报告。

        【绝对规则】
        1.  所有分析必须100%基于我提供的数据，不得使用任何你自己的知识库
        2.  不得编造任何我没有提供的数据
        3.  语言简洁，重点突出，适合在手机上阅读
        4.  报告日期是: {today}

        {us_market_summary}

        【前一个交易日港股数据】
        {data_table}

        【报告结构】
        1.  隔夜美股影响：简要说明美股表现对今天港股开盘的可能影响
        2.  昨日市场回顾：总结前一个交易日的整体市场情况
        3.  昨日强势股：列出涨幅前5名的股票
        4.  昨日弱势股：列出跌幅前5名的股票
        5.  今日重点关注：列出RSI超买、超卖和成交量异常的股票
        6.  免责声明

        请严格按照以上结构生成报告。
        """
    else:
        prompt = f"""
        你是一个严格遵守事实的港股分析师。现在是香港时间下午4点30分，港股刚刚收盘。请基于我提供的当天完整交易数据，生成一份收盘总结报告。

        【绝对规则】
        1.  所有分析必须100%基于我提供的数据，不得使用任何你自己的知识库
        2.  不得编造任何我没有提供的数据
        3.  每一个结论都必须明确引用对应的指标
        4.  语言简洁，重点突出，适合在手机上阅读
        5.  报告日期是: {today}

        【今日港股数据】
        {data_table}

        【报告结构】
        1.  市场概览：简要说明今天整体市场的涨跌情况和主要特征
        2.  涨幅前5名：列出涨幅最大的5只股票，结合成交量和趋势进行简要分析
        3.  跌幅前5名：列出跌幅最大的5只股票，结合成交量和趋势进行简要分析
        4.  特别关注：列出RSI超买、超卖、成交量异常和高股息的股票
        5.  明日展望：基于今天的数据，简要说明明天可能的市场走势
        6.  免责声明

        请严格按照以上结构生成报告。
        """
    
    ai_result = generate_content_with_retry(prompt)
    
    if ai_result:
        return ai_result
    else:
        return generate_python_analysis(all_stocks, is_morning, us_data)

# ------------------
# 改进版Telegram发送函数
# ------------------
def send_telegram(message):
    MAX_CHUNK_SIZE = 3800
    
    chunks = []
    while len(message) > MAX_CHUNK_SIZE:
        split_index = message.rfind("\n", 0, MAX_CHUNK_SIZE)
        if split_index == -1:
            split_index = MAX_CHUNK_SIZE
        chunks.append(message[:split_index])
        message = message[split_index:]
    chunks.append(message)
    
    print(f"📤 报告分为 {len(chunks)} 条消息发送")
    
    for i, chunk in enumerate(chunks):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        try:
            response = requests.post(url, data=data, timeout=15)
            response.raise_for_status()
            print(f"✅ 第 {i+1}/{len(chunks)} 条消息发送成功")
            time.sleep(1)
        except Exception as e:
            print(f"❌ 第 {i+1}/{len(chunks)} 条消息发送失败 (Markdown): {e}")
            print("🔄 尝试用纯文本发送...")
            data["parse_mode"] = None
            try:
                response = requests.post(url, data=data, timeout=15)
                response.raise_for_status()
                print(f"✅ 第 {i+1}/{len(chunks)} 条消息发送成功 (纯文本)")
                time.sleep(1)
            except Exception as e2:
                print(f"❌ 纯文本发送也失败: {e2}")

# ------------------
# Main
# ------------------
if __name__ == "__main__":
    # 自动检测当前运行模式
    now = datetime.now()
    is_morning = now.hour < 12  # 12点之前都是早盘模式
    
    print(f"📅 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔄 运行模式: {'早盘模式' if is_morning else '收盘模式'}")
    
    print("📥 开始获取股票数据和计算技术指标...")
    all_stocks = get_all_stock_data()
    
    us_data = None
    if is_morning:
        print("📥 获取隔夜美股数据...")
        us_data = get_us_market_data()
    
    print("🤖 开始分析...")
    report = generate_full_report(all_stocks, is_morning, us_data)
    
    print("📤 发送报告到Telegram...")
    send_telegram(report)
    
    print("✅ 运行完成！")
    print("\n报告内容：")
    print(report)

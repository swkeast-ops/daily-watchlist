import os
import requests
import time
import yfinance as yf
import pandas as pd
import numpy as np
import google.genai as genai
from google.genai import types
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ------------------
# 配置香港时区
# ------------------
HK_TZ = ZoneInfo("Asia/Hong_Kong")

# ------------------
# Load environment secrets
# ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STOCK_LIST = os.getenv("STOCK_LIST")
FORCE_MODE = os.getenv("FORCE_MODE", None)

stock_list = [x.strip() for x in STOCK_LIST.split(",")]

# ------------------
# 智能数据验证系统
# ------------------
def validate_and_clean_data(stock_data):
    """验证并清理所有数据，确保数值在合理范围内"""
    cleaned = stock_data.copy()
    issues = []
    
    # 1. 智能股息率检测和修正
    if isinstance(cleaned["dividend_yield"], (int, float)):
        if cleaned["dividend_yield"] > 1:
            # yfinance返回的是已经乘以100的百分比，不需要再乘
            cleaned["dividend_yield"] = round(cleaned["dividend_yield"], 2)
            issues.append(f"股息率格式修正: {stock_data['dividend_yield']} → {cleaned['dividend_yield']}%")
        else:
            # 正常小数，乘以100
            cleaned["dividend_yield"] = round(cleaned["dividend_yield"] * 100, 2)
    
    # 2. 合理性检查
    # 股息率合理范围：0-20%
    if isinstance(cleaned["dividend_yield"], (int, float)) and (cleaned["dividend_yield"] < 0 or cleaned["dividend_yield"] > 20):
        issues.append(f"股息率异常: {cleaned['dividend_yield']}%，标记为N/A")
        cleaned["dividend_yield"] = "N/A"
    
    # PE合理范围：0-100
    if isinstance(cleaned["pe_ratio"], (int, float)) and (cleaned["pe_ratio"] < 0 or cleaned["pe_ratio"] > 100):
        issues.append(f"PE异常: {cleaned['pe_ratio']}，标记为N/A")
        cleaned["pe_ratio"] = "N/A"
    
    # PB合理范围：0-20
    if isinstance(cleaned["pb_ratio"], (int, float)) and (cleaned["pb_ratio"] < 0 or cleaned["pb_ratio"] > 20):
        issues.append(f"PB异常: {cleaned['pb_ratio']}，标记为N/A")
        cleaned["pb_ratio"] = "N/A"
    
    # ROE合理范围：-50-50%
    if isinstance(cleaned["roe"], (int, float)) and (cleaned["roe"] < -50 or cleaned["roe"] > 50):
        issues.append(f"ROE异常: {cleaned['roe']}%，标记为N/A")
        cleaned["roe"] = "N/A"
    
    # 涨跌幅合理范围：-30-30%
    if isinstance(cleaned["change_pct"], (int, float)) and (cleaned["change_pct"] < -30 or cleaned["change_pct"] > 30):
        issues.append(f"涨跌幅异常: {cleaned['change_pct']}%，标记为N/A")
        cleaned["change_pct"] = "N/A"
    
    # 成交量变化合理范围：-90-500%
    if isinstance(cleaned["volume_change_pct"], (int, float)) and (cleaned["volume_change_pct"] < -90 or cleaned["volume_change_pct"] > 500):
        issues.append(f"成交量变化异常: {cleaned['volume_change_pct']}%，标记为N/A")
        cleaned["volume_change_pct"] = "N/A"
    
    return cleaned, issues

# ------------------
# Gemini AI setup with auto-retry and fallback
# ------------------
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_content_with_retry(prompt, max_retries=2):
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
# Get ALL stock data (带数据验证)
# ------------------
def get_all_stock_data():
    all_data = []
    total_issues = 0
    
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
            dividend_yield = info.get("dividendYield", 0)
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
            raw_data = {
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
            
            # 验证和清理数据
            cleaned_data, issues = validate_and_clean_data(raw_data)
            total_issues += len(issues)
            
            if issues:
                print(f"⚠️ {symbol} 数据问题: {issues}")
            
            all_data.append(cleaned_data)
            
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
    
    print(f"✅ 数据获取完成，共处理 {len(all_data)} 只股票，发现 {total_issues} 个数据问题并修正")
    return all_data

# ------------------
# 纯Python智能分析引擎
# ------------------
def generate_python_analysis(all_stocks, is_morning, us_data=None, data_date=None):
    valid_stocks = [s for s in all_stocks if s["price"] != "N/A"]
    
    # 按涨跌幅排序
    sorted_stocks = sorted(valid_stocks, key=lambda x: x["change_pct"], reverse=True)
    top_gainers = sorted_stocks[:5]
    top_losers = sorted_stocks[-5:]
    
    # 计算市场整体情况
    avg_change = round(np.mean([s["change_pct"] for s in valid_stocks]), 2)
    up_count = len([s for s in valid_stocks if s["change_pct"] > 0])
    down_count = len([s for s in valid_stocks if s["change_pct"] < 0])
    
    # 筛选特别关注股票（优化后的筛选条件）
    overbought = [s for s in valid_stocks if s["rsi"] != "N/A" and s["rsi"] > 70]
    oversold = [s for s in valid_stocks if s["rsi"] != "N/A" and s["rsi"] < 30]
    # 只筛选成交量变化>200%或<-70%的真正异常
    high_volume = [s for s in valid_stocks if s["volume_change_pct"] != "N/A" and (s["volume_change_pct"] > 200 or s["volume_change_pct"] < -70)]
    high_dividend = [s for s in valid_stocks if s["dividend_yield"] != "N/A" and s["dividend_yield"] > 5]
    
    # 生成报告
    if is_morning:
        report = f"🌅 港股早盘观察报告\n\n"
        report += f"📅 数据日期: {data_date.strftime('%Y年%m月%d日')}\n"
        report += f"⏰ 生成时间: {datetime.now(HK_TZ).strftime('%Y年%m月%d日 %H:%M')}\n\n"
        
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
            report += "成交量异常: " + ", ".join([s["code"] for s in high_volume]) + "\n"
        if high_dividend:
            report += "高股息(>5%): " + ", ".join([s["code"] for s in high_dividend]) + "\n"
        
    else:
        report = f"🌇 港股收盘总结报告\n\n"
        report += f"📅 数据日期: {data_date.strftime('%Y年%m月%d日')}\n"
        report += f"⏰ 生成时间: {datetime.now(HK_TZ).strftime('%Y年%m月%d日 %H:%M')}\n\n"
        
        report += "📈 今日市场概览\n"
        report += f"整体平均涨跌幅: {avg_change}%\n"
        report += f"上涨股票: {up_count} 只 | 下跌股票: {down_count} 只\n\n"
        
        report += "🚀 今日涨幅前5名\n"
        for stock in top_gainers:
            comment = ""
            if stock["volume_change_pct"] != "N/A" and stock["volume_change_pct"] > 100:
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
            if stock["volume_change_pct"] != "N/A" and stock["volume_change_pct"] > 100:
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
            report += "成交量异常: " + ", ".join([s["code"] for s in high_volume]) + "\n"
        if high_dividend:
            report += "高股息(>5%): " + ", ".join([s["code"] for s in high_dividend]) + "\n"
    
    report += "\n⚠️ 分析仅供参考，不构成投资建议。"
    
    return report

# ------------------
# 终极防幻觉AI分析
# ------------------
def generate_full_report(all_stocks, is_morning, us_data=None, data_date=None):
    valid_stocks = [s for s in all_stocks if s["price"] != "N/A"]
    
    # 按涨跌幅排序
    sorted_stocks = sorted(valid_stocks, key=lambda x: x["change_pct"], reverse=True)
    top_gainers = sorted_stocks[:5]
    top_losers = sorted_stocks[-5:]
    
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
        你是一个严格遵守事实的港股分析师。现在是香港时间早上9点，港股还没有开盘。请基于我提供的{data_date.strftime('%Y年%m月%d日')}的收盘数据和隔夜美股数据，生成一份早盘观察报告。

        【绝对规则】
        1.  所有分析必须100%基于我提供的数据，不得使用任何你自己的知识库
        2.  不得编造任何我没有提供的数据
        3.  语言简洁，重点突出，适合在手机上阅读

        {us_market_summary}

        【{data_date.strftime('%Y年%m月%d日')}港股数据】
        {data_table}

        【报告结构】
        1.  隔夜美股影响：简要说明美股表现对今天港股开盘的可能影响
        2.  昨日市场回顾：总结{data_date.strftime('%Y年%m月%d日')}的整体市场情况
        3.  昨日强势股：列出涨幅前5名的股票
        4.  昨日弱势股：列出跌幅前5名的股票
        5.  今日重点关注：列出RSI超买、超卖和成交量异常的股票
        6.  免责声明

        请严格按照以上结构生成报告。
        """
    else:
        prompt = f"""
        你是一个严格遵守事实的港股分析师。现在是香港时间下午4点30分，港股刚刚收盘。请基于我提供的{data_date.strftime('%Y年%m月%d日')}的完整交易数据，生成一份收盘总结报告。

        【绝对规则】
        1.  所有分析必须100%基于我提供的数据，不得使用任何你自己的知识库
        2.  不得编造任何我没有提供的数据
        3.  每一个结论都必须明确引用对应的指标
        4.  语言简洁，重点突出，适合在手机上阅读

        【{data_date.strftime('%Y年%m月%d日')}港股数据】
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
        # 给AI生成的报告加上日期和时间
        header = f"🌅 港股早盘观察报告\n\n" if is_morning else f"🌇 港股收盘总结报告\n\n"
        header += f"📅 数据日期: {data_date.strftime('%Y年%m月%d日')}\n"
        header += f"⏰ 生成时间: {datetime.now(HK_TZ).strftime('%Y年%m月%d日 %H:%M')}\n\n"
        return header + ai_result
    else:
        return generate_python_analysis(all_stocks, is_morning, us_data, data_date)

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
    # 获取当前香港时间
    now_hk = datetime.now(HK_TZ)
    print(f"📅 当前香港时间: {now_hk.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 判断运行模式
    if FORCE_MODE == "morning":
        is_morning = True
        print("🔄 强制运行模式: 早盘模式")
    elif FORCE_MODE == "evening":
        is_morning = False
        print("🔄 强制运行模式: 收盘模式")
    else:
        # 自动判断：12点之前是早盘模式，12点之后是收盘模式
        is_morning = now_hk.hour < 12
        print(f"🔄 自动运行模式: {'早盘模式' if is_morning else '收盘模式'}")
    
    # 确定数据日期
    if is_morning:
        # 早盘模式使用前一个交易日的数据
        data_date = now_hk - timedelta(days=1)
        # 如果是周一，使用上周五的数据
        if data_date.weekday() >= 5:
            data_date = data_date - timedelta(days=data_date.weekday() - 4)
    else:
        # 收盘模式使用当天的数据
        data_date = now_hk
    
    print(f"📅 数据日期: {data_date.strftime('%Y年%m月%d日')}")
    
    print("📥 开始获取股票数据和计算技术指标...")
    all_stocks = get_all_stock_data()
    
    us_data = None
    if is_morning:
        print("📥 获取隔夜美股数据...")
        us_data = get_us_market_data()
    
    print("🤖 开始分析...")
    report = generate_full_report(all_stocks, is_morning, us_data, data_date)
    
    print("📤 发送报告到Telegram...")
    send_telegram(report)
    
    print("✅ 运行完成！")
    print("\n报告内容：")
    print(report)

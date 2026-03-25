from openai import OpenAI
import json

# 👇 关键：加 base_url
client = OpenAI(
    api_key="sk-ee021bca039c4d76940d735f4bbfe2a4",
    base_url="https://api.deepseek.com"
)

def analyze_sentiment(text):
    prompt = f"""
你是一个专业的情感分析助手，请对以下文本进行分析：

文本：{text}

请输出JSON格式：
{{
    "sentiment": "正面/中性/负面",
    "score": 0-1之间的小数,
    "reason": "简要解释原因"
}}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",  # 👈 改这里
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    result = response.choices[0].message.content

    try:
        return json.loads(result)
    except:
        return {"error": "解析失败", "raw": result}


if __name__ == "__main__":
    text = "这个APP卡死了，体验太差了"
    result = analyze_sentiment(text)
    print(result)
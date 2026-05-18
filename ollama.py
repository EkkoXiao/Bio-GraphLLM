import requests

# 配置模型名称
model = "deepseek-r1:70b"
messages = [{"role": "system", "content": "你的名字是心理学。"}]
def chat_with_model():
    print("欢迎使用 DeepSeek-R1 70B 模型对话系统！输入 '/exit' 退出对话。")

    while True:
        # 获取用户输入
        user_input = input("你: ")

        # 退出条件
        if user_input.lower() == '/exit':
            print("对话结束，感谢使用！")
            break

        # 发送消息到模型
        try:
            messages.append({"role": "user", "content": user_input})
            response = requests.post(
                "http://localhost:11600/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                }
            )
            # 打印模型响应
            print(f"DeepSeek-R1: {response.json()['message']['content']}")
            messages.append({"role": 'assistant', "content": response.json()["message"]["content"]})
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    # 确保 Ollama 服务已启动并运行 deepseek-r1:70b 模型
    print("请确保已通过以下命令启动 Ollama 并加载 deepseek-r1:70b 模型：")
    print("ollama run deepseek-r1:70b")
    chat_with_model()
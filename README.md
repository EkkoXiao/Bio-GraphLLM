## Step1 Ollama 启动 (可选，不启动则 generate 等接口无法使用)

```
tmux new-session -s ollama
ollama serve

tmux new-session -s deepseek
ollama run deepseek-r1:70b
```
其中 tmux 可以新建其他进程，如果需要启动多个大模型（如 QWEN）需要指定暴露在其他端口。

## Step2 FastAPI 挂载

```
tmux attach -t api
python fastapi_server.py

tmux attach -t ngrok
./ngrok http 8000 (回到根目录执行这句话)
```

获得 ngrok 的链接后，将其替换 `streamlit_app.py` 中的 API_URL。
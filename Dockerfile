# 1. 基础镜像
FROM python:3.10-slim

# 2. 安装系统依赖（opencv 需要 libxcb 等库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3. 设置工作目录
WORKDIR /app

# 4. 复制并安装 Python 依赖
COPY requirements.txt .
RUN mkdir -p /root/.config/pip && \
    echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > /root/.config/pip/pip.conf
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制项目代码
COPY . .

# 6. 暴露端口
EXPOSE 8000

# 7. 启动
CMD ["python", "app.py"]

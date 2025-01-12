# 构建阶段
FROM 10.130.0.9/hub/python:3.10-slim as builder

WORKDIR /app
COPY requirements-prod.txt .

# 创建虚拟环境
# --no-cache-dir 禁用缓存可以减少 Docker 镜像的大小
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements-prod.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行阶段
FROM 10.130.0.9/hub/python:3.10-slim

WORKDIR /app

# 复制虚拟环境和应用代码
COPY --from=builder /opt/venv /opt/venv
COPY . .

# 设置环境变量
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 19990

# 使用生产配置启动应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "19990"]
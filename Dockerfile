# 使用官方 Python 基础镜像
FROM 10.130.0.9/hub/python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . .


# 安装 Python 依赖
RUN pip install --upgrade pip
# --no-cache-dir 禁用缓存可以减少 Docker 镜像的大小
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# 如果有环境变量配置文件，请确保它被正确加载
RUN pip install python-dotenv

# 暴露应用端口
EXPOSE 19990

# 设置默认命令来启动 FastAPI 应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "19990", "--reload"]
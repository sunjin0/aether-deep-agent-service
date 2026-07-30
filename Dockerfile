# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# 依赖层仅依赖锁文件；业务源码改动时不会重新解析或下载 Python 依赖。
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen --no-install-project

# 复制源码后仅安装当前项目本身，第三方依赖复用上一层及 uv 下载缓存。
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8010
CMD ["aether-deep-agent-service", "--host", "0.0.0.0", "--port", "8010"]

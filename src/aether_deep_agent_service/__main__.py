import argparse

import uvicorn


def main() -> None:
    """解析监听参数并启动 Uvicorn 服务。"""
    parser = argparse.ArgumentParser(description="Run the Aether Deep Agent service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8010, type=int)
    args = parser.parse_args()
    uvicorn.run("aether_deep_agent_service.app:app", host=args.host, port=args.port)

"""使测试与开发环境 .env 隔离。

pydantic-settings 的 Settings 默认读取仓库根目录的 .env；若开发环境配置了
真实的 callback_base_url / model / MCP 地址，测试会发起真实网络请求并可能
在 Windows 上挂起。这里把相关环境变量显式置空，覆盖 .env 的取值。
"""

import os

# AETHER_DEEP_AGENT_DATABASE_URL 不置空：app.py 模块级 build_application() 会用
# get_settings() 构造 RunStore，空 URL 会导致导入失败。
_HERMETIC_ENV = {
    "AETHER_DEEP_AGENT_CALLBACK_BASE_URL": "",
    "AETHER_DEEP_AGENT_MODEL": "",
    "AETHER_DEEP_AGENT_MCP_URL": "",
    "AETHER_DEEP_AGENT_SHARED_SECRET": "",
    "OPENAI_API_KEY": "",
    "OPENAI_BASE_URL": "",
    "ANTHROPIC_API_KEY": "",
    "GOOGLE_API_KEY": "",
}

for _key, _value in _HERMETIC_ENV.items():
    os.environ.setdefault(_key, _value)
    if os.environ.get(_key) != _value:
        os.environ[_key] = _value

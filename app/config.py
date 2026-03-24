import os

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


# 数据库配置
MYSQL_HOST = _require_env("MYSQL_HOST")
MYSQL_PORT = int(_require_env("MYSQL_PORT"))
MYSQL_USER = _require_env("MYSQL_USER")
MYSQL_PASSWORD = _require_env("MYSQL_PASSWORD")
MYSQL_DATABASE = _require_env("MYSQL_DATABASE")
MYSQL_CHARSET = os.getenv("MYSQL_CHARSET", "utf8mb4")

# 文件上传目录
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

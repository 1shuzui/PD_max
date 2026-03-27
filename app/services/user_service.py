"""
用户模块服务层
负责用户注册、登录、密码加密及 JWT Token 签发
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app import config
from app.database import get_conn

logger = logging.getLogger(__name__)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ==================== 密码 / JWT 工具 ====================

def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def create_access_token(payload: Dict[str, Any]) -> str:
    data = payload.copy()
    data["exp"] = datetime.now(timezone.utc) + timedelta(
        minutes=config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(data, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    except JWTError:
        return None


# ==================== 内部工具 ====================

_USER_COLS = "id, username, real_name, role, phone, email, is_active, created_at"


def _row_to_dict(cur, row) -> Dict[str, Any]:
    cols = [d[0] for d in cur.description]
    u = dict(zip(cols, row))
    if isinstance(u.get("created_at"), datetime):
        u["created_at"] = u["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    return u


# ==================== UserService ====================

class UserService:

    # ---------- A1 登录 ----------

    def login(self, username: str, password: str) -> Dict[str, Any]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_USER_COLS}, hashed_password FROM users WHERE username = %s",
                    (username,),
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("账号或密码错误")
                user = _row_to_dict(cur, row)

        if not user["is_active"]:
            raise ValueError("该账户已被禁用，请联系管理员")
        if not verify_password(password, user.pop("hashed_password")):
            raise ValueError("账号或密码错误")

        token = create_access_token({"sub": str(user["id"]), "username": user["username"], "role": user["role"]})
        logger.info(f"用户登录成功: id={user['id']}, username={username}")
        return {"code": 200, "msg": "登录成功", "token": token, "user": user}

    # ---------- A2 获取用户列表 ----------

    def list_users(
        self,
        keyword: Optional[str] = None,
        role: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        conditions = ["is_active = 1"]
        params: List[Any] = []

        if keyword:
            conditions.append("(username LIKE %s OR real_name LIKE %s OR phone LIKE %s)")
            like = f"%{keyword}%"
            params += [like, like, like]
        if role:
            conditions.append("role = %s")
            params.append(role)

        where = " AND ".join(conditions)
        offset = (page - 1) * page_size

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM users WHERE {where}", params)
                total = cur.fetchone()[0]

                cur.execute(
                    f"SELECT {_USER_COLS} FROM users WHERE {where} "
                    f"ORDER BY id LIMIT %s OFFSET %s",
                    params + [page_size, offset],
                )
                rows = cur.fetchall()
                user_list = [_row_to_dict(cur, r) for r in rows]

        return {"code": 200, "data": {"total": total, "list": user_list}}

    # ---------- A3 新增用户 ----------

    def create_user(
        self,
        username: str,
        password: str,
        real_name: Optional[str] = None,
        role: str = "user",
        phone: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cur.fetchone():
                    raise ValueError("账号已存在")

                cur.execute(
                    "INSERT INTO users (username, hashed_password, real_name, role, phone, email) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (username, hash_password(password), real_name, role, phone, email),
                )
                new_id = cur.lastrowid

        logger.info(f"新用户创建: id={new_id}, username={username}")
        return {"code": 200, "msg": "用户创建成功", "id": new_id}

    # ---------- A4 修改角色 ----------

    def update_role(self, user_id: int, role: str) -> Dict[str, Any]:
        if role not in ("admin", "user"):
            raise ValueError("角色值无效，只允许 admin 或 user")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE id = %s AND is_active = 1", (user_id,))
                if not cur.fetchone():
                    raise ValueError(f"用户 id={user_id} 不存在")
                cur.execute("UPDATE users SET role = %s WHERE id = %s", (role, user_id))
        return {"code": 200, "msg": "角色修改成功"}

    # ---------- A5 修改密码 ----------

    def change_password(self, user_id: int, admin_key: str, new_password: str) -> Dict[str, Any]:
        if admin_key != config.JWT_SECRET_KEY:
            raise PermissionError("管理员密钥错误")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE id = %s AND is_active = 1", (user_id,))
                if not cur.fetchone():
                    raise ValueError(f"用户 id={user_id} 不存在")
                cur.execute(
                    "UPDATE users SET hashed_password = %s WHERE id = %s",
                    (hash_password(new_password), user_id),
                )
        return {"code": 200, "msg": "密码修改成功"}

    # ---------- A6 删除用户（软删除） ----------

    def delete_user(self, user_id: int, current_user_id: int) -> Dict[str, Any]:
        if user_id == current_user_id:
            raise ValueError("不可删除当前登录账号")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE id = %s AND is_active = 1", (user_id,))
                if not cur.fetchone():
                    raise ValueError(f"用户 id={user_id} 不存在")
                cur.execute("UPDATE users SET is_active = 0 WHERE id = %s", (user_id,))
        return {"code": 200, "msg": "用户已删除"}


# ==================== 单例工厂 ====================

_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service

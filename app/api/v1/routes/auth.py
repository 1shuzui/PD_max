"""
用户认证模块路由
接口前缀：/auth
包含接口：
  A1. POST /auth/login           - 登录，返回 JWT token
  A2. GET  /auth/users           - 获取用户列表（仅 admin）
  A3. POST /auth/users           - 新增用户（仅 admin）
  A4. POST /auth/update_role     - 修改用户角色（仅 admin）
  A5. POST /auth/change_password - 修改用户密码
  A6. POST /auth/delete_user     - 删除用户（仅 admin，软删除）
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from app.services.user_service import UserService, get_user_service, decode_access_token

router = APIRouter(prefix="/auth", tags=["用户认证"])

_bearer = HTTPBearer()


def _current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="未登录或token已过期")
    return payload


def _require_admin(user: dict = Depends(_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    return user


# ==================== 请求体 ====================

class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    real_name: Optional[str] = None
    password: str = Field(..., min_length=6)
    role: str = "user"
    phone: Optional[str] = None
    email: Optional[str] = None


class UpdateRoleRequest(BaseModel):
    id: int
    role: str


class ChangePasswordRequest(BaseModel):
    id: int
    admin_key: str
    new_password: str = Field(..., min_length=6)


class DeleteUserRequest(BaseModel):
    id: int


# ==================== 路由 ====================

# A1 登录
@router.post("/login", summary="用户登录")
def login(body: LoginRequest, service: UserService = Depends(get_user_service)):
    try:
        return service.login(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# A2 获取用户列表
@router.get("/users", summary="获取用户列表（仅admin）")
def list_users(
    keyword: Optional[str] = None,
    role: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    _: dict = Depends(_require_admin),
    service: UserService = Depends(get_user_service),
):
    try:
        return service.list_users(keyword=keyword, role=role, page=page, page_size=page_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# A3 新增用户
@router.post("/users", summary="新增用户（仅admin）")
def create_user(
    body: CreateUserRequest,
    _: dict = Depends(_require_admin),
    service: UserService = Depends(get_user_service),
):
    try:
        return service.create_user(
            username=body.username,
            password=body.password,
            real_name=body.real_name,
            role=body.role,
            phone=body.phone,
            email=body.email,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# A4 修改用户角色
@router.post("/update_role", summary="修改用户角色（仅admin）")
def update_role(
    body: UpdateRoleRequest,
    _: dict = Depends(_require_admin),
    service: UserService = Depends(get_user_service),
):
    try:
        return service.update_role(body.id, body.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# A5 修改密码
@router.post("/change_password", summary="修改用户密码")
def change_password(
    body: ChangePasswordRequest,
    service: UserService = Depends(get_user_service),
):
    try:
        return service.change_password(body.id, body.admin_key, body.new_password)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# A6 删除用户
@router.post("/delete_user", summary="删除用户（仅admin，软删除）")
def delete_user(
    body: DeleteUserRequest,
    current: dict = Depends(_require_admin),
    service: UserService = Depends(get_user_service),
):
    try:
        return service.delete_user(body.id, current_user_id=int(current["sub"]))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

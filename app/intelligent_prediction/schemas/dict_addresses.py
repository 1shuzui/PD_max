"""TL 字典仓库 / 冶炼厂地址查询（预测模块用）。"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TlDictEntityAddress(BaseModel):
    """``dict_warehouses`` 或 ``dict_factories`` 单条记录的地址信息。"""

    id: int
    name: str
    province: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    address: Optional[str] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    full_address: str = Field(
        default="",
        description="省、市、区、详细地址按顺序拼接，便于前端一行展示",
    )


class WarehouseSmelterAddressLookupResponse(BaseModel):
    """按名称从主库字典解析出的仓库与冶炼厂地址；未命中字段为 null。"""

    warehouse: Optional[TlDictEntityAddress] = None
    smelter: Optional[TlDictEntityAddress] = None

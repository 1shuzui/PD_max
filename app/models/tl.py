from typing import List

from pydantic import BaseModel, Field


class ComparisonRequest(BaseModel):
    """接口4 请求体"""
    选中仓库id列表: List[int] = Field(..., description="选中的仓库ID列表")
    冶炼厂id列表: List[int] = Field(..., description="冶炼厂ID列表")
    品类id列表: List[int] = Field(..., description="品类ID列表")


class UploadFreightRequest(BaseModel):
    """接口6 请求体（单条）"""
    仓库: str = Field(..., description="仓库名称，如 北京仓")
    冶炼厂: str = Field(..., description="冶炼厂名称，如 华北冶炼厂")
    运费: float = Field(..., description="运费金额（元/吨）")


class UpdateCategoryMappingRequest(BaseModel):
    """接口7 请求体"""
    品类id: int = Field(..., description="品类分组ID")
    品类名称: List[str] = Field(..., description="品类名称列表，第一个为主名称")


class ConfirmPriceTableItem(BaseModel):
    """确认价格表 - 单条明细"""
    冶炼厂id: int = Field(..., description="冶炼厂ID")
    品类id: int = Field(..., description="品类分组ID")
    价格: float = Field(..., description="单价（元/吨）")
    原始品类名: str = Field("", description="OCR识别的原始品类名")


class ConfirmPriceTableRequest(BaseModel):
    """接口5b 请求体 - 确认写入报价数据"""
    报价日期: str = Field(..., description="报价日期，格式 YYYY-MM-DD")
    仓库id: int = Field(..., description="发货仓库ID")
    数据: List[ConfirmPriceTableItem] = Field(..., description="报价明细列表")

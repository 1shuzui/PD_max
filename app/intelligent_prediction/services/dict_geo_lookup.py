"""从 TL 字典表解析仓库 / 冶炼厂所在「市」（dict_warehouses、dict_factories）。

匹配顺序：精确名称 → 去空白后全等 → 子串模糊（字典名含导入名，或导入名含字典名），
多命中时按规则打分取一条。
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from app.database import get_conn
from app.intelligent_prediction.logging_utils import get_logger

logger = get_logger(__name__)

_TABLE_WH = "dict_warehouses"
_TABLE_DF = "dict_factories"
_DictTable = Literal["dict_warehouses", "dict_factories"]


def _compact(s: str) -> str:
    """去掉首尾与中间空白（含全角空格），用于宽松等值比较。"""
    return re.sub(r"[\s\u3000]+", "", (s or "").strip())


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def _rank_match(query: str, dict_name: str) -> tuple[int, int]:
    """分数越小越好；第二项用于同档排序。"""
    q, n = query.strip(), dict_name.strip()
    if not q:
        return (99, 0)
    if n == q:
        return (0, 0)
    qc, nc = _compact(q), _compact(n)
    if qc and nc == qc:
        return (1, 0)
    # 字典名是导入名的子串：如导入「上海宝钢一号库」命中字典「宝钢」
    if n and n in q:
        return (2, -len(n))
    # 导入名是字典名的子串：如导入「宝钢」命中「上海宝钢股份有限公司」
    if q and q in n:
        return (3, len(n))
    return (9, len(n))


def _city_from_row(row: tuple) -> Optional[str]:
    if not row or not row[0]:
        return None
    t = str(row[0]).strip()
    return t or None


def _lookup_city_one_table(cur, table: _DictTable, raw_name: str) -> Optional[str]:
    assert table in (_TABLE_WH, _TABLE_DF)
    name = (raw_name or "").strip()
    if not name:
        return None

    cur.execute(
        f"SELECT city FROM {table} WHERE name = %s AND is_active = 1 LIMIT 1",
        (name,),
    )
    row = cur.fetchone()
    c = _city_from_row(row)
    if c:
        return c

    compact = _compact(name)
    if len(compact) < 2:
        return None

    esc = _escape_like(name)
    like_pat = f"%{esc}%"
    # 与 _compact 一致：去掉空格与全角空格、常见换行
    cur.execute(
        f"""
        SELECT city, name FROM {table}
        WHERE is_active = 1
          AND city IS NOT NULL AND TRIM(city) <> ''
          AND (
            REPLACE(REPLACE(REPLACE(REPLACE(TRIM(name), CHAR(10), ''), CHAR(13), ''), ' ', ''), '　', '') = %s
            OR name LIKE %s ESCAPE '\\\\'
            OR %s LIKE CONCAT('%%', name, '%%')
          )
        LIMIT 80
        """,
        (compact, like_pat, name),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    valid = [(c, n) for c, n in rows if c is not None and n is not None and str(n).strip()]
    if not valid:
        return None

    city_cell, dict_name = min(
        valid,
        key=lambda r: (_rank_match(name, str(r[1])), str(r[1])),
    )
    return _city_from_row((city_cell,))


def lookup_warehouse_factory_cities(
    warehouse_name: str,
    smelter_name: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """返回 (仓库所在市, 冶炼厂所在市)。查不到则为 None。

    匹配规则（每条内按顺序，命中即停；模糊阶段多行取打分最优）：

    1. ``name`` 与导入名**完全一致**（trim），且 ``is_active = 1``；
    2. 去掉空白后 ``name`` 与导入名全等；
    3. ``name LIKE %导入%`` 或 ``导入 LIKE CONCAT('%', name, '%')``；
       多命中时：精确 > 去空白等 > 字典名为导入子串（偏好更长字典名）
       > 导入名为字典子串（偏好更短字典名）。
    """
    wh_city: Optional[str] = None
    sm_city: Optional[str] = None
    wn = (warehouse_name or "").strip()
    sn = (smelter_name or "").strip() or None
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if wn:
                    wh_city = _lookup_city_one_table(cur, _TABLE_WH, wn)
                if sn:
                    sm_city = _lookup_city_one_table(cur, _TABLE_DF, sn)
    except Exception:
        logger.exception("dict geo lookup failed warehouse=%s smelter=%s", wn, sn)
        return None, None
    return wh_city, sm_city


def _nullable_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _full_address(
    province: Any, city: Any, district: Any, address: Any
) -> str:
    parts: list[str] = []
    for p in (province, city, district, address):
        s = _nullable_str(p)
        if s:
            parts.append(s)
    return "".join(parts)


def _row_to_address_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    """将 dict_warehouses / dict_factories 一行转为 API 字典。"""
    _id, n, prov, city, dist, addr, lon, lat = row

    def _fnum(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    return {
        "id": int(_id),
        "name": str(n).strip(),
        "province": _nullable_str(prov),
        "city": _nullable_str(city),
        "district": _nullable_str(dist),
        "address": _nullable_str(addr),
        "longitude": _fnum(lon),
        "latitude": _fnum(lat),
        "full_address": _full_address(prov, city, dist, addr),
    }


def _lookup_address_one_table(cur, table: _DictTable, raw_name: str) -> Optional[dict[str, Any]]:
    """按与「市」解析相同的名称匹配规则，取一条字典记录的完整地址字段。"""
    assert table in (_TABLE_WH, _TABLE_DF)
    name = (raw_name or "").strip()
    if not name:
        return None

    cur.execute(
        f"""
        SELECT id, name, province, city, district, address, longitude, latitude
        FROM {table}
        WHERE name = %s AND is_active = 1
        LIMIT 1
        """,
        (name,),
    )
    row = cur.fetchone()
    if row:
        return _row_to_address_dict(row)

    compact = _compact(name)
    if len(compact) < 2:
        return None

    esc = _escape_like(name)
    like_pat = f"%{esc}%"
    cur.execute(
        f"""
        SELECT id, name, province, city, district, address, longitude, latitude
        FROM {table}
        WHERE is_active = 1
          AND (
            REPLACE(REPLACE(REPLACE(REPLACE(TRIM(name), CHAR(10), ''), CHAR(13), ''), ' ', ''), '　', '') = %s
            OR name LIKE %s ESCAPE '\\\\'
            OR %s LIKE CONCAT('%%', name, '%%')
          )
        LIMIT 80
        """,
        (compact, like_pat, name),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    valid = [r for r in rows if r and len(r) > 1 and str(r[1]).strip()]
    if not valid:
        return None

    best = min(
        valid,
        key=lambda r: (_rank_match(name, str(r[1])), str(r[1])),
    )
    return _row_to_address_dict(best)


def lookup_warehouse_smelter_dict_addresses(
    warehouse_name: str,
    smelter_name: Optional[str],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """从 TL 主库 ``dict_warehouses`` / ``dict_factories`` 解析仓库与冶炼厂地址（名称匹配规则同 :func:`lookup_warehouse_factory_cities`）。

    返回两个字典，字段含 ``id``、``name``、省市区、``address``、经纬度及拼接的 ``full_address``；未命中则为 ``None``。
    """
    wh_out: Optional[dict[str, Any]] = None
    sm_out: Optional[dict[str, Any]] = None
    wn = (warehouse_name or "").strip()
    sn = (smelter_name or "").strip() or None
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if wn:
                    wh_out = _lookup_address_one_table(cur, _TABLE_WH, wn)
                if sn:
                    sm_out = _lookup_address_one_table(cur, _TABLE_DF, sn)
    except Exception:
        logger.exception(
            "dict address lookup failed warehouse=%s smelter=%s", wn, sn
        )
        return None, None
    return wh_out, sm_out

"""
TL比价模块服务层
负责仓库、冶炼厂、品类、比价、运费、价格表、品类映射等数据库操作
"""
import hashlib
import logging
import os
import uuid
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import UPLOAD_DIR
from app.database import get_conn
from battery_quote_service1 import BatteryQuoteService

logger = logging.getLogger(__name__)

PRICE_TABLE_UPLOAD_DIR = Path(UPLOAD_DIR) / "price_tables"
PRICE_TABLE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class TLService:

    # ==================== 接口1：获取仓库列表 ====================

    def get_warehouses(self) -> List[Dict[str, Any]]:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id AS `仓库id`, name AS `仓库名` "
                        "FROM dict_warehouses "
                        "WHERE is_active = 1 "
                        "ORDER BY id"
                    )
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"获取仓库列表失败: {e}")
            raise

    # ==================== 接口2：获取冶炼厂列表 ====================

    def get_smelters(self) -> List[Dict[str, Any]]:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id AS `冶炼厂id`, name AS `冶炼厂` "
                        "FROM dict_factories "
                        "WHERE is_active = 1 "
                        "ORDER BY id"
                    )
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"获取冶炼厂列表失败: {e}")
            raise

    # ==================== 接口3：获取品类列表 ====================

    def get_categories(self) -> List[Dict[str, Any]]:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT category_id AS `品类id`, "
                        "GROUP_CONCAT(name ORDER BY row_id SEPARATOR '、') AS `品类名` "
                        "FROM dict_categories "
                        "WHERE is_active = 1 "
                        "GROUP BY category_id "
                        "ORDER BY category_id"
                    )
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"获取品类列表失败: {e}")
            raise

    # ==================== 接口4：获取比价表 ====================

    def get_comparison(
        self,
        warehouse_ids: List[int],
        smelter_ids: List[int],
        category_ids: List[int],
    ) -> List[Dict[str, Any]]:
        if not warehouse_ids or not smelter_ids or not category_ids:
            return []

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    wh_placeholders = ",".join(["%s"] * len(warehouse_ids))
                    sm_placeholders = ",".join(["%s"] * len(smelter_ids))
                    cat_placeholders = ",".join(["%s"] * len(category_ids))

                    # 获取每个品类的主名称（is_main=1），若无则取任意一条
                    cur.execute(
                        f"SELECT DISTINCT category_id, "
                        f"COALESCE(MAX(CASE WHEN is_main=1 THEN name END), MAX(name)) AS cat_name "
                        f"FROM dict_categories "
                        f"WHERE category_id IN ({cat_placeholders}) AND is_active = 1 "
                        f"GROUP BY category_id",
                        tuple(category_ids),
                    )
                    cat_map: Dict[int, str] = {
                        row[0]: row[1] for row in cur.fetchall()
                    }

                    # 查询运费：取最新生效日期
                    sql = f"""
                        SELECT
                            dw.id        AS warehouse_id,
                            dw.name      AS warehouse_name,
                            df.id        AS factory_id,
                            df.name      AS factory_name,
                            fr.price_per_ton AS freight
                        FROM freight_rates fr
                        JOIN dict_warehouses dw ON fr.warehouse_id = dw.id
                        JOIN dict_factories  df ON fr.factory_id  = df.id
                        WHERE dw.id IN ({wh_placeholders})
                          AND df.id IN ({sm_placeholders})
                          AND fr.effective_date = (
                              SELECT MAX(fr2.effective_date)
                              FROM freight_rates fr2
                              WHERE fr2.factory_id  = fr.factory_id
                                AND fr2.warehouse_id = fr.warehouse_id
                          )
                    """
                    cur.execute(sql, tuple(warehouse_ids) + tuple(smelter_ids))
                    freight_rows = cur.fetchall()

                    # 笛卡尔积：运费记录 × 品类
                    result = []
                    for wid, wname, fid, fname, freight in freight_rows:
                        for cid in category_ids:
                            cat_name = cat_map.get(cid)
                            if cat_name is None:
                                continue
                            result.append({
                                "仓库": wname,
                                "冶炼厂": fname,
                                "品类": cat_name,
                                "运费列表": float(freight) if freight is not None else None,
                            })
                    return result

        except Exception as e:
            logger.error(f"获取比价表失败: {e}")
            raise

    # ==================== 接口5：上传价格表（OCR解析） ====================

    def _match_factory(
        self, ocr_name: str, factory_list: List[Tuple[int, str]]
    ) -> Optional[int]:
        """将 OCR 识别出的工厂名匹配到 dict_factories 中的冶炼厂，返回 factory_id"""
        if not ocr_name or ocr_name == "未知工厂":
            return None
        for fid, fname in factory_list:
            # 双向包含匹配
            if fname in ocr_name or ocr_name in fname:
                return fid
        return None

    def _match_category(
        self, ocr_cat: str, category_list: List[Tuple[int, int, str]]
    ) -> Optional[Tuple[int, int]]:
        """将 OCR 识别出的品类名匹配到 dict_categories，返回 (category_id, row_id)"""
        if not ocr_cat:
            return None
        for row_id, cat_id, cname in category_list:
            if cname in ocr_cat or ocr_cat in cname:
                return (cat_id, row_id)
        return None

    def upload_price_table(self, files: List[Any]) -> Dict[str, Any]:
        saved_paths: List[Tuple[str, str, str]] = []  # (save_path, md5, original_filename)
        try:
            # 1. 保存图片到磁盘
            for upload_file in files:
                content = upload_file.file.read()
                md5 = hashlib.md5(content).hexdigest()
                suffix = Path(upload_file.filename).suffix or ".jpg"
                filename = f"{uuid.uuid4().hex}{suffix}"
                save_path = PRICE_TABLE_UPLOAD_DIR / filename

                with open(save_path, "wb") as f:
                    f.write(content)
                saved_paths.append((str(save_path), md5, upload_file.filename))

            # 2. 从数据库加载冶炼厂和品类字典
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, name FROM dict_factories WHERE is_active = 1"
                    )
                    factory_list: List[Tuple[int, str]] = list(cur.fetchall())

                    cur.execute(
                        "SELECT row_id, category_id, name "
                        "FROM dict_categories WHERE is_active = 1"
                    )
                    category_list: List[Tuple[int, int, str]] = list(cur.fetchall())

            # 3. 对每张图片做 OCR 解析 + 匹配
            ocr_service = BatteryQuoteService()

            # parsed: {factory_id: {category_id: price}}
            parsed: Dict[int, Dict[int, float]] = {}
            unmatched_factories: List[str] = []
            unmatched_categories: List[str] = []
            details: List[Dict[str, Any]] = []

            for image_path, md5, orig_name in saved_paths:
                ocr_result = ocr_service.parse_image(image_path)

                if ocr_result.get("error"):
                    details.append({
                        "image": orig_name,
                        "error": ocr_result["error"],
                    })
                    continue

                factory_name_ocr = ocr_result.get("factory", "未知工厂")
                factory_id = self._match_factory(factory_name_ocr, factory_list)

                if factory_id is None and factory_name_ocr != "未知工厂":
                    if factory_name_ocr not in unmatched_factories:
                        unmatched_factories.append(factory_name_ocr)

                image_detail: Dict[str, Any] = {
                    "image": orig_name,
                    "factory_name": factory_name_ocr,
                    "factory_id": factory_id,
                    "date": ocr_result.get("date"),
                    "items": [],
                }

                for item in ocr_result.get("items", []):
                    raw_cat = item["category"]
                    price = item["price"]

                    match = self._match_category(raw_cat, category_list)
                    if match:
                        cat_id, row_id = match
                    else:
                        cat_id, row_id = None, None
                        if raw_cat not in unmatched_categories:
                            unmatched_categories.append(raw_cat)

                    image_detail["items"].append({
                        "raw_category_name": raw_cat,
                        "category_id": cat_id,
                        "price": price,
                    })

                    # 汇总到 parsed 结构
                    if factory_id is not None and cat_id is not None:
                        parsed.setdefault(factory_id, {})[cat_id] = price

                details.append(image_detail)

            return {
                "code": 200,
                "data": {
                    "parsed": parsed,
                    "unmatched": {
                        "factories": unmatched_factories,
                        "categories": unmatched_categories,
                    },
                    "details": details,
                },
            }

        except Exception as e:
            logger.error(f"上传价格表失败: {e}")
            for path, _, _ in saved_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise

    # ==================== 接口5b：确认价格表写入数据库 ====================

    def confirm_price_table(
        self,
        quote_date_str: str,
        warehouse_id: int,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        前端确认后，将报价数据写入 quote_orders + quote_details。
        items 格式: [{"冶炼厂id": 1, "品类id": 3, "价格": 9350, "原始品类名": "电动车"}, ...]
        """
        if not items:
            raise ValueError("报价数据不能为空")

        try:
            quote_dt = date.fromisoformat(quote_date_str)
        except (ValueError, TypeError):
            raise ValueError(f"日期格式不正确: {quote_date_str}，应为 YYYY-MM-DD")

        batch_no = uuid.uuid4().hex[:16]

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    # 插入主单
                    cur.execute(
                        "INSERT INTO quote_orders "
                        "(quote_date, upload_batch_no, warehouse_id, status) "
                        "VALUES (%s, %s, %s, 'DRAFT')",
                        (quote_dt, batch_no, warehouse_id),
                    )
                    order_id = cur.lastrowid

                    # 插入明细
                    for item in items:
                        factory_id = item["冶炼厂id"]
                        category_id = item.get("品类id")
                        price = item["价格"]
                        raw_name = item.get("原始品类名", "")

                        # 查找 mapped_category_row_id
                        mapped_row_id = None
                        if category_id is not None:
                            cur.execute(
                                "SELECT row_id FROM dict_categories "
                                "WHERE category_id = %s AND is_active = 1 "
                                "ORDER BY is_main DESC LIMIT 1",
                                (category_id,),
                            )
                            row = cur.fetchone()
                            if row:
                                mapped_row_id = row[0]

                        cur.execute(
                            "INSERT INTO quote_details "
                            "(order_id, factory_id, raw_category_name, "
                            "mapped_category_row_id, category_id, weight_tons, unit_price) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (
                                order_id,
                                factory_id,
                                raw_name,
                                mapped_row_id,
                                category_id,
                                0,  # weight_tons 暂时为 0，后续可补充
                                price,
                            ),
                        )

            return {
                "code": 200,
                "msg": "报价数据已写入数据库",
                "order_id": order_id,
                "batch_no": batch_no,
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"确认价格表写入失败: {e}")
            raise

    # ==================== 接口6：上传运费 ====================

    def upload_freight(self, freight_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    today = date.today().isoformat()
                    for item in freight_list:
                        warehouse_name = item["仓库"]
                        smelter_name = item["冶炼厂"]
                        freight = item["运费"]

                        cur.execute(
                            "SELECT id FROM dict_warehouses WHERE name = %s AND is_active = 1",
                            (warehouse_name,),
                        )
                        wh_row = cur.fetchone()
                        if not wh_row:
                            raise ValueError(f"仓库 '{warehouse_name}' 不存在或未启用")

                        cur.execute(
                            "SELECT id FROM dict_factories WHERE name = %s AND is_active = 1",
                            (smelter_name,),
                        )
                        sm_row = cur.fetchone()
                        if not sm_row:
                            raise ValueError(f"冶炼厂 '{smelter_name}' 不存在或未启用")

                        cur.execute(
                            "INSERT INTO freight_rates "
                            "(factory_id, warehouse_id, price_per_ton, effective_date) "
                            "VALUES (%s, %s, %s, %s) "
                            "ON DUPLICATE KEY UPDATE "
                            "price_per_ton = VALUES(price_per_ton), "
                            "updated_at = CURRENT_TIMESTAMP",
                            (sm_row[0], wh_row[0], freight, today),
                        )
            return {"code": 200, "msg": "运费数据已存入数据库"}

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"上传运费失败: {e}")
            raise

    # ==================== 接口7：更新品类映射表 ====================

    def update_category_mapping(
        self,
        category_id: int,
        names: List[str],
    ) -> Dict[str, Any]:
        if not names:
            raise ValueError("品类名称列表不能为空")

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    # 将该 category_id 下所有旧记录的 is_main 置为 0
                    cur.execute(
                        "UPDATE dict_categories SET is_main = 0 WHERE category_id = %s",
                        (category_id,),
                    )

                    for i, name in enumerate(names):
                        is_main = 1 if i == 0 else 0

                        cur.execute(
                            "SELECT row_id, category_id FROM dict_categories WHERE name = %s",
                            (name,),
                        )
                        existing = cur.fetchone()

                        if existing:
                            cur.execute(
                                "UPDATE dict_categories "
                                "SET category_id = %s, is_main = %s, is_active = 1 "
                                "WHERE row_id = %s",
                                (category_id, is_main, existing[0]),
                            )
                        else:
                            category_code = f"CAT_{name.upper()[:10]}_{uuid.uuid4().hex[:6]}"
                            cur.execute(
                                "INSERT INTO dict_categories "
                                "(category_id, category_code, name, is_main, is_active) "
                                "VALUES (%s, %s, %s, %s, 1)",
                                (category_id, category_code, name, is_main),
                            )

            return {"code": 200, "msg": "品类映射表更新成功，数据已存入数据库"}

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"更新品类映射失败: {e}")
            raise


# ==================== 单例工厂 ====================

_tl_service: Optional[TLService] = None


def get_tl_service() -> TLService:
    global _tl_service
    if _tl_service is None:
        _tl_service = TLService()
    return _tl_service

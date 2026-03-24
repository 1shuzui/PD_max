"""测试 OCR 识图功能：用 1.jpg 验证 BatteryQuoteService 解析结果"""
import json
from battery_quote_service1 import BatteryQuoteService

service = BatteryQuoteService()
result = service.parse_image("1.jpg")

print(json.dumps(result, ensure_ascii=False, indent=2))

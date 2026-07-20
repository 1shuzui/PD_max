# 图片检测模块

该目录提供凭证图片篡改检测、人工反馈二审、候选模型训练和历史导出能力。生产检测使用 v3；v4 是独立的离线局部取证候选模型，不会自动影响线上结果。

## 分层与调用链

```text
HTTP /ai-detection/api/v3/detect
  -> routes/ai_detection.py
  -> workflows/inference_v3.py
  -> core/ OCR、ROI、特征与像素取证
  -> services/ 上传、历史、反馈、模型注册
```

| 目录 | 责任 |
| --- | --- |
| `core/` | 无状态的图像特征、OCR token、金额候选、ROI、时间和语义计算。不得写数据库或发起 HTTP 请求。 |
| `workflows/` | v3 推理、v3 训练、v4 离线训练、候选评估和数据集重测编排。 |
| `services/` | 上传暂存、历史、反馈二审、训练任务、模型注册和规则检测等持久化与业务操作。 |
| `runtime/` | 配置路径、CPU 线程限制、EasyOCR 资产与下载补丁。 |

根目录不再提供旧模块转发文件。代码必须从上述分层路径导入，例如：

```python
from app.ai_detection.workflows.inference_v3 import InferenceEngineAPI
from app.ai_detection.services.feedback_manager import FeedbackManager
from app.ai_detection.core.ocr_utils import run_full_image_ocr
```

## 生产检测

v3 自动流程为：读取原图 -> EasyOCR -> 提取金额、姓名、时间 ROI -> 对每个 ROI 推理 -> 文档规则覆盖 -> 聚合为“正常 / 可疑 / 篡改 / 无法自动检测”。没有关键 ROI 时必须返回“无法自动检测”，不能回退为整图硬判正常或篡改。

接口前缀为 `/ai-detection`。批量检测应先使用 `POST /api/v3/upload` 完成全部上传，再以 `task_id` 调用 `POST /api/v3/detect`，最后轮询 `GET /api/v3/result/{task_id}`。标注图使用 `GET /api/v3/result/{task_id}/visualization`。

相关运行环境变量：

| 变量 | 作用 |
| --- | --- |
| `AI_DETECTION_ENABLED` | 为 `0` 时不注册鉴伪路由。 |
| `AI_DETECTION_PRELOAD` | 为真时在应用启动阶段预加载 OCR 和模型；默认按首次请求加载。 |
| `AI_MAX_CONCURRENT_TASKS` | 控制真实模型推理并发；生产默认应保持 `1`。 |

## 数据、模型与结果包

- `images/normal/` 与 `images/tampered/` 是规范原图；`images/derived/` 只保存派生样本，不计入独立评测。
- `pptest` 标记为训练回放样本，不可作为独立泛化指标。
- 初审反馈保留在 `feedback/`；只有二审确认后的样本可作为训练来源。篡改二审必须包含至少一个金额、姓名或时间区域。
- ROI 标注在 `locate_json/annotations/legacy/` 与 `locate_json/annotations/v4/`；已有标注图在 `locate_json/visualizations/legacy/`。
- 模型版本保存于 `models/versions/<version>/`，活跃版本由 `models/registry.json` 决定；候选模型不得覆盖或自动切换活跃模型。
- v3 特征缓存使用 `models/cache/v3_global/`；非训练重测结果写入 `models/evaluations/v3/<run_id>/`。结果包应包含 JSON、CSV、报告、图表、标注图和 SHA-256 清单。

修改 `images/`、`feedback/`、模型注册表、模型文件或训练阈值前，必须先取得明确确认。SHA-256 仅用于去重、审计和数据泄漏防护，不得作为生产推理的篡改旁路。

## 开发与验证

在项目根目录执行：

```bash
.venv/bin/python -m py_compile $(find app scripts tests -name '*.py' -print)
.venv/bin/python -m unittest discover -s tests -p 'test_ai_detection*.py' -v
.venv/bin/python scripts/retest_v3_dataset.py
```

最后一条命令只读当前活跃 v3 模型和数据集，结果保存在 `models/evaluations/v3/`。涉及应用启动或部署时，还应验证：

```bash
curl http://127.0.0.1:<PORT>/ai-detection/api/v3/health
```

健康端点在按需加载模式下可能先显示 OCR 和模型未加载；提交一张异步检测任务后，应显示 `ocr_available=true` 和 `global_model_loaded=true`。

## 代码规范

新增或修改的注释必须使用中文，只说明不直观的业务约束、性能边界或数据安全原因。保持 `core/` 无副作用；不要在 ROI 循环中加载模型、OCR 或执行垃圾回收；长批次结束后由工作流统一释放临时缓存。

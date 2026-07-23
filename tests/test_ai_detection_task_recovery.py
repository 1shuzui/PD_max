# -*- coding: utf-8 -*-
import tempfile
import unittest
import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from fastapi import UploadFile

from app.ai_detection.services.history_export import render_annotated_jpeg
from app.ai_detection.core.amount_candidates import OCRToken
from app.api.v1.routes.ai_detection import (
    DetectionDomainServiceV3,
    MemoryTaskRegistry,
    STORAGE_DIR,
    TaskRecordDTO,
    TaskStatusEnum,
    _persist_upload_task,
    _task_sidecar_path,
    build_task_record_from_persistence,
)


class TaskRecoveryTests(unittest.TestCase):
    def test_build_task_record_from_storage_after_restart(self):
        task_id = "115e4ba8-e2bc-41c1-9ec8-c9cd91f0e1bf"
        storage_path = STORAGE_DIR / f"{task_id}.jpg"
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            storage_path.write_bytes(b"\xff\xd8\xff")
            with patch(
                "app.api.v1.routes.ai_detection.get_async_v3_history_by_task_id",
                return_value=None,
            ):
                task = build_task_record_from_persistence(task_id)
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual(task.status, TaskStatusEnum.FAILED)
            self.assertIn("中断", task.error_msg or "")
        finally:
            if storage_path.is_file():
                storage_path.unlink()
            sidecar = _task_sidecar_path(task_id)
            if sidecar.is_file():
                sidecar.unlink()

    def test_build_task_record_from_upload_sidecar_after_restart(self):
        async def run_case():
            task_id = "115e4ba8-e2bc-41c1-9ec8-c9cd91f1bf"
            storage_path = STORAGE_DIR / f"{task_id}.jpg"
            sidecar = _task_sidecar_path(task_id)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                storage_path.write_bytes(b"\xff\xd8\xff")
                registry = MemoryTaskRegistry()
                await registry.create_task(
                    task_id=task_id,
                    image_path=str(storage_path),
                    original_filename="receipt.jpg",
                    image_created_at="2026-07-06 10:00:00",
                    batch="20260706001",
                )
                registry._store.clear()
                with patch(
                    "app.api.v1.routes.ai_detection.get_async_v3_history_by_task_id",
                    return_value=None,
                ):
                    task = build_task_record_from_persistence(task_id)
                self.assertIsNotNone(task)
                assert task is not None
                self.assertEqual(task.status, TaskStatusEnum.UPLOADED)
                self.assertEqual(task.original_filename, "receipt.jpg")
                self.assertEqual(task.image_created_at, "2026-07-06 10:00:00")
                self.assertEqual(task.batch, "20260706001")
            finally:
                if storage_path.is_file():
                    storage_path.unlink()
                if sidecar.is_file():
                    sidecar.unlink()

        asyncio.run(run_case())

    def test_pending_sidecar_after_restart_is_reported_interrupted(self):
        async def run_case():
            task_id = "pending-after-restart"
            storage_path = STORAGE_DIR / f"{task_id}.jpg"
            sidecar = _task_sidecar_path(task_id)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_path.write_bytes(b"\xff\xd8\xff")
            registry = MemoryTaskRegistry()
            try:
                await registry.create_task(task_id, str(storage_path), "pending.jpg")
                await registry.update_task(task_id, status=TaskStatusEnum.PENDING)
                registry._store.clear()
                with patch(
                    "app.api.v1.routes.ai_detection.get_async_v3_history_by_task_id",
                    return_value=None,
                ):
                    task = build_task_record_from_persistence(task_id)
                self.assertIsNotNone(task)
                assert task is not None
                self.assertEqual(task.status, TaskStatusEnum.FAILED)
                self.assertIn("中断", task.error_msg or "")
            finally:
                storage_path.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)

        asyncio.run(run_case())

    def test_cancel_terminal_history_does_not_delete_archived_image(self):
        async def run_case():
            from app.api.v1.routes.ai_detection import cancel_task

            task_id = "completed-history-cancel"
            with tempfile.TemporaryDirectory() as tmp:
                archived = Path(tmp) / "123.png"
                archived.write_bytes(b"archived")
                completed = TaskRecordDTO(
                    task_id=task_id,
                    status=TaskStatusEnum.COMPLETED,
                    created_at="2026-07-14T00:00:00",
                    image_path=str(archived),
                    original_filename="done.png",
                    result={"result": "正常"},
                )
                registry = MemoryTaskRegistry()
                with patch(
                    "app.api.v1.routes.ai_detection.build_task_record_from_persistence",
                    return_value=completed,
                ):
                    result = await cancel_task(task_id, registry)
                self.assertEqual(result["status"], "already_finished")
                self.assertTrue(archived.is_file())

        asyncio.run(run_case())

    def test_delete_uploaded_task_removes_image_and_sidecar(self):
        async def run_case():
            task_id = "delete-sidecar-task"
            storage_path = STORAGE_DIR / f"{task_id}.jpg"
            sidecar = _task_sidecar_path(task_id)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_path.write_bytes(b"\xff\xd8\xff")
            registry = MemoryTaskRegistry()
            await registry.create_task(
                task_id=task_id,
                image_path=str(storage_path),
                original_filename="receipt.jpg",
                batch="20260706002",
            )
            self.assertTrue(sidecar.is_file())

            removed = await registry.delete_task(task_id)

            self.assertTrue(removed)
            self.assertFalse(storage_path.exists())
            self.assertFalse(sidecar.exists())

        asyncio.run(run_case())

    def test_persist_upload_task_only_creates_uploaded_task(self):
        async def run_case():
            registry = MemoryTaskRegistry()
            ok, encoded = cv2.imencode(".png", np.full((8, 10, 3), 127, dtype=np.uint8))
            self.assertTrue(ok)
            payload = encoded.tobytes()
            upload = UploadFile(io.BytesIO(payload), filename="付款截图.jpg")
            task = await _persist_upload_task(
                file=upload,
                registry=registry,
                image_created_at="2026-07-06 11:00:00",
                batch="20260706003",
            )
            try:
                self.assertEqual(task.status, TaskStatusEnum.UPLOADED)
                self.assertEqual(task.original_filename, "付款截图.jpg")
                self.assertEqual(task.image_created_at, "2026-07-06 11:00:00")
                self.assertEqual(task.batch, "20260706003")
                self.assertEqual(task.media_type, "image/png")
                self.assertEqual(task.size_bytes, len(payload))
                self.assertEqual(len(task.content_sha256 or ""), 64)
                self.assertEqual(Path(task.image_path or "").suffix, ".png")
                self.assertEqual(Path(task.image_path or "").read_bytes(), payload)
                self.assertTrue(_task_sidecar_path(task.task_id).is_file())
            finally:
                await registry.delete_task(task.task_id)

        asyncio.run(run_case())

    def test_finalize_completed_task_clears_ephemeral_storage_after_history_archive(self):
        async def run_case():
            task_id = "cleanup-completed-task"
            storage_path = STORAGE_DIR / f"{task_id}.jpg"
            sidecar = _task_sidecar_path(task_id)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_path.write_bytes(b"\xff\xd8\xff")
            registry = MemoryTaskRegistry()
            await registry.create_task(
                task_id=task_id,
                image_path=str(storage_path),
                original_filename="receipt.jpg",
                batch="20260706004",
            )
            service = DetectionDomainServiceV3(registry, asyncio.Semaphore(1))
            with patch.object(service, "_persist_history", new=AsyncMock()):
                with patch(
                    "app.api.v1.routes.ai_detection.get_async_v3_history_by_task_id",
                    return_value={"id": 321, "status": "COMPLETED"},
                ):
                    with patch(
                        "app.api.v1.routes.ai_detection.get_ai_detection_history_image_path",
                        return_value=Path("/tmp/ai_detection_history_images/321.jpg"),
                    ):
                        await service._finalize_completed_task(
                            task_id,
                            str(storage_path),
                            original_filename="receipt.jpg",
                            bbox=None,
                            result={"result": "正常", "confidence": 0.1},
                            multi_results=[{"result": "正常", "confidence": 0.1}],
                            image_created_at="2026-07-06 11:00:00",
                            batch="20260706004",
                        )

            self.assertFalse(storage_path.exists())
            self.assertFalse(sidecar.exists())
            self.assertIsNone(await registry.get_task(task_id))

        asyncio.run(run_case())

    def test_build_task_record_from_completed_history(self):
        task_id = "test-task-completed"
        content_sha256 = "a" * 64
        with patch(
            "app.api.v1.routes.ai_detection.get_async_v3_history_by_task_id",
            return_value={
                "task_id": task_id,
                "status": "COMPLETED",
                "created_at": "2026-06-01 17:00:00",
                "image_created_at": "2026-05-31 12:34:56",
                "batch": "codex-history-batch-001",
                "original_filename": "chatgptedit5.png",
                "content_sha256": content_sha256,
                "size_bytes": 12345,
                "media_type": "image/png",
                "bbox": None,
                "outcome": {
                    "result": {"result": "正常", "confidence": 0.2},
                    "linked_rule_checks": {"status": "正常", "available": True},
                },
            },
        ):
            with patch(
                "app.api.v1.routes.ai_detection.get_rule_checks_history_by_task_id",
                return_value=None,
            ):
                task = build_task_record_from_persistence(task_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.status, TaskStatusEnum.COMPLETED)
        self.assertEqual(task.result.get("result"), "正常")
        self.assertEqual(task.image_created_at, "2026-05-31 12:34:56")
        self.assertEqual(task.batch, "codex-history-batch-001")
        self.assertEqual(task.content_sha256, content_sha256)
        self.assertEqual(task.size_bytes, 12345)
        self.assertEqual(task.media_type, "image/png")

    def test_persist_history_copies_upload_metadata_from_task(self):
        async def run_case():
            registry = MemoryTaskRegistry()
            task_id = "persist-upload-meta"
            storage_path = STORAGE_DIR / f"{task_id}.png"
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_path.write_bytes(b"png")
            await registry.create_task(
                task_id=task_id,
                image_path=str(storage_path),
                original_filename="中文原名.png",
                content_sha256="b" * 64,
                size_bytes=3,
                media_type="image/png",
            )
            service = DetectionDomainServiceV3(registry, asyncio.Semaphore(1))
            try:
                with patch(
                    "app.api.v1.routes.ai_detection.insert_ai_detection_history"
                ) as insert_history:
                    await service._persist_history(
                        task_id=task_id,
                        original_filename="中文原名.png",
                        bbox=None,
                        status="COMPLETED",
                        result={"result": "正常"},
                        source_image_path=str(storage_path),
                    )

                kwargs = insert_history.call_args.kwargs
                self.assertEqual(kwargs["content_sha256"], "b" * 64)
                self.assertEqual(kwargs["size_bytes"], 3)
                self.assertEqual(kwargs["media_type"], "image/png")
                self.assertEqual(
                    kwargs["outcome"]["upload_meta"]["original_filename"],
                    "中文原名.png",
                )
            finally:
                await registry.delete_task(task_id)

        asyncio.run(run_case())

    def test_assign_region_numbers_overwrites_sorted_order(self):
        rows = DetectionDomainServiceV3._assign_region_numbers(
            [
                {"result": "篡改", "region_no": 99, "field_label": "金额"},
                {"result": "正常", "field_label": "姓名"},
            ]
        )

        self.assertEqual([row["region_no"] for row in rows], [1, 2])

    def test_v3_suspicious_is_preserved_by_default(self):
        source = {
            "result": "可疑",
            "confidence": 0.596,
            "reason": "全局UI布局异常",
        }

        result = DetectionDomainServiceV3._resolve_v3_suspicious_result(source)

        self.assertIs(result, source)
        self.assertEqual(result["result"], "可疑")
        self.assertNotIn("v3_suspicious_resolved", result)

    def test_v3_suspicious_below_decision_threshold_resolves_to_normal(self):
        with patch("app.api.v1.routes.ai_detection.V3_RESOLVE_SUSPICIOUS_RESULTS", True):
            result = DetectionDomainServiceV3._resolve_v3_suspicious_result(
                {
                    "result": "可疑",
                    "confidence": 0.585,
                    "reason": "全局UI布局异常",
                }
            )

        self.assertEqual(result["result"], "正常")
        self.assertTrue(result["v3_suspicious_resolved"])
        self.assertIn("未达到自动篡改阈值", result["reason"])

    def test_v3_suspicious_above_decision_threshold_resolves_to_tampered(self):
        with patch("app.api.v1.routes.ai_detection.V3_RESOLVE_SUSPICIOUS_RESULTS", True):
            result = DetectionDomainServiceV3._resolve_v3_suspicious_result(
                {
                    "result": "可疑",
                    "confidence": 0.596,
                    "reason": "全局UI布局异常",
                }
            )

        self.assertEqual(result["result"], "篡改")
        self.assertTrue(result["v3_suspicious_resolved"])
        self.assertIn("达到自动篡改阈值", result["reason"])

    def test_execute_async_without_key_regions_returns_unable_to_detect(self):
        async def run_case():
            registry = MemoryTaskRegistry()
            task_id = "no-key-region-task"
            with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                await registry.create_task(
                    task_id=task_id,
                    image_path=tmp.name,
                    original_filename="receipt.jpg",
                )
                service = DetectionDomainServiceV3(registry, asyncio.Semaphore(1))
                service._cached_key_rois = []

                with patch("app.api.v1.routes.ai_detection.ensure_ai_detection_runtime", new=AsyncMock()):
                    with patch("app.api.v1.routes.ai_detection.EngineContainer.instance", object()):
                        with patch("app.api.v1.routes.ai_detection.EngineContainer.ocr_reader", object()):
                            with patch.object(service, "_run_ocr_once", return_value=None):
                                with patch.object(service, "_finalize_completed_task") as finalize:
                                    await service.execute_async(task_id, tmp.name, None)

                finalize.assert_called_once()
                result = finalize.call_args.kwargs["result"]
                self.assertEqual(result["result"], "无法自动检测")
                self.assertIn("金额、姓名、时间", result["reason"])
                self.assertEqual(finalize.call_args.kwargs["persist_bbox"]["note"], "no_key_field_regions")

        asyncio.run(run_case())

    def test_execute_async_waits_for_shared_ai_work_lock(self):
        async def run_case():
            registry = MemoryTaskRegistry()
            task_id = "shared-work-lock-task"
            image_path = STORAGE_DIR / f"{task_id}.jpg"
            image_path.write_bytes(b"image")
            await registry.create_task(task_id, str(image_path), "lock.jpg")
            service = DetectionDomainServiceV3(registry, asyncio.Semaphore(1))
            work_lock = asyncio.Lock()

            from app.api.v1.routes.ai_detection import EngineContainer

            old_lock = EngineContainer.work_lock
            EngineContainer.work_lock = work_lock
            entered = asyncio.Event()

            async def fake_locked(*_args, **_kwargs):
                entered.set()

            try:
                with patch.object(service, "_execute_async_locked", side_effect=fake_locked):
                    await work_lock.acquire()
                    task = asyncio.create_task(service.execute_async(task_id, str(image_path)))
                    await asyncio.sleep(0)
                    self.assertFalse(entered.is_set())
                    work_lock.release()
                    await task
                    self.assertTrue(entered.is_set())
            finally:
                EngineContainer.work_lock = old_lock
                if work_lock.locked():
                    work_lock.release()
                await registry.delete_task(task_id)

        asyncio.run(run_case())

    def test_large_document_without_key_roi_does_not_hard_label_tampered(self):
        registry = MemoryTaskRegistry()
        service = DetectionDomainServiceV3(registry, asyncio.Semaphore(1))
        image = np.full((2600, 3000, 3), 245, dtype=np.uint8)

        left, right = 180, 2700
        top, bottom = 360, 1420
        for y in range(top, bottom + 1, 140):
            cv2.line(image, (left, y), (right, y), (0, 0, 0), 5)
        for x in (left, 760, 1120, right):
            cv2.line(image, (x, top), (x, bottom), (0, 0, 0), 5)
        cv2.circle(image, (2200, 2100), 230, (0, 0, 210), 28)
        service._cached_img_cv2 = image

        result = service._visual_document_override()

        self.assertIsNone(result)

    def test_doubao_watermark_override_is_direct_tamper_result(self):
        service = DetectionDomainServiceV3(MemoryTaskRegistry(), asyncio.Semaphore(1))
        service._cached_tokens = [
            OCRToken("豆包AI生成", "豆包AI生成", (820, 900, 922, 918), 0.96, 102, 18, 909.0),
        ]

        result = service._doubao_watermark_override()

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["result"], "篡改")
        self.assertEqual(result["field_label"], "AI水印")
        self.assertEqual(result["evidence_type"], "ai_generated_document")
        self.assertTrue(result["hard_tamper_flags"]["doubao_ai_watermark"])

    def test_render_annotated_jpeg_draws_region_number_labels(self):
        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            cv2.imencode(".jpg", image)[1].tofile(tmp.name)
            outcome = {
                "multi_results": [
                    {
                        "result": "正常",
                        "confidence": 0.1,
                        "bbox": [20, 20, 50, 40],
                        "original_bbox": [20, 20, 70, 60],
                        "region_no": 1,
                        "field_label": "金额",
                    },
                    {
                        "result": "篡改",
                        "confidence": 0.9,
                        "bbox": [90, 20, 50, 40],
                        "original_bbox": [90, 20, 140, 60],
                        "region_no": 2,
                        "field_label": "姓名",
                    },
                ]
            }

            with patch("app.ai_detection.services.history_export.load_chinese_font") as mock_font:
                from PIL import ImageFont

                mock_font.return_value = ImageFont.load_default()
                jpeg = render_annotated_jpeg(Path(tmp.name), outcome)

        rendered = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(rendered)
        assert rendered is not None
        self.assertLess(float(np.mean(rendered[24:48, 24:48])), 245.0)
        self.assertLess(float(np.mean(rendered[24:48, 94:118])), 245.0)


if __name__ == "__main__":
    unittest.main()

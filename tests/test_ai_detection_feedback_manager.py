import json
import io
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from app.ai_detection.services.feedback_manager import FeedbackEntryReviewedError, FeedbackManager
from app.ai_detection.services.reviewed_dataset import ReviewRegionRequired


class FeedbackManagerTests(unittest.TestCase):
    def _manager(self, tmp: str) -> FeedbackManager:
        cfg = {
            "feedback": {
                "storage_dir": str(Path(tmp) / "feedback"),
            }
        }
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        return FeedbackManager(str(cfg_path))

    def _seed_entry(self, manager: FeedbackManager, judgment: str = "suspicious") -> dict:
        src = manager.base_dir / "source.jpg"
        Image.new("RGB", (8, 6), (120, 80, 40)).save(src, format="JPEG")
        return manager.save_judgment(
            task_id="task-1",
            judgment=judgment,
            image_path=str(src),
            bbox=[1, 2, 3, 4],
            result={"result": "篡改"},
            note="note",
            original_filename="付款 截图.jpg",
            initial_reviewer="user-7",
        )

    def test_list_entries_includes_folder_name_and_image_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            entry = self._seed_entry(manager, "wrong")

            rows = manager.list_entries("wrong")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["entry_id"], entry["entry_id"])
            self.assertTrue(rows[0]["folder_name"])
            self.assertEqual(rows[0]["judgment"], "wrong")
            self.assertIn("/api/v3/feedback/", rows[0]["image_url"])
            self.assertEqual(rows[0]["original_filename"], "付款 截图.jpg")
            self.assertEqual(rows[0]["review_status"], "pending")

    def test_update_entry_moves_between_judgments(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            entry = self._seed_entry(manager, "suspicious")
            folder = manager.list_entries("suspicious")[0]["folder_name"]

            updated = manager.update_entry(folder, "wrong")

            self.assertIsNotNone(updated)
            self.assertEqual(updated["judgment"], "wrong")
            self.assertFalse((manager.suspicious_dir / folder).exists())
            self.assertTrue((manager.wrong_dir / folder).exists())

            metadata = json.loads((manager.wrong_dir / folder / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["judgment"], "wrong")

    def test_update_entry_can_change_display_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            self._seed_entry(manager, "correct")
            folder = manager.list_entries("correct")[0]["folder_name"]

            updated = manager.update_entry(
                folder,
                "correct",
                original_filename="二审确认的付款截图.png",
            )

            self.assertIsNotNone(updated)
            self.assertEqual(updated["original_filename"], "二审确认的付款截图.png")

    def test_delete_entry_removes_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            self._seed_entry(manager, "correct")
            folder = manager.list_entries("correct")[0]["folder_name"]

            removed = manager.delete_entry(folder)

            self.assertTrue(removed)
            self.assertIsNone(manager.get_entry(folder))

    def test_second_review_creates_reviewed_sample_and_blocks_direct_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            self._seed_entry(manager, "wrong")
            folder = manager.list_entries("wrong")[0]["folder_name"]

            reviewed = manager.review_entry(
                folder,
                label=0,
                reviewer="admin-1",
                note="确认正常",
            )

            self.assertEqual(reviewed["review_status"], "reviewed")
            self.assertEqual(reviewed["true_label"], 0)
            self.assertTrue(reviewed["reviewed_sample_id"])
            sample = manager.reviewed.get_entry(reviewed["reviewed_sample_id"])
            self.assertIsNotNone(sample)
            with self.assertRaises(FeedbackEntryReviewedError):
                manager.delete_entry(folder)

    def test_revoke_second_review_removes_last_training_copy_and_allows_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            self._seed_entry(manager, "wrong")
            folder = manager.list_entries("wrong")[0]["folder_name"]
            reviewed = manager.review_entry(
                folder,
                label=1,
                reviewer="admin-1",
                regions=[{"field_type": "amount", "x1": 0.1, "y1": 0.1, "x2": 0.8, "y2": 0.5}],
            )
            sample_id = reviewed["reviewed_sample_id"]

            reverted = manager.revoke_review(folder, reviewer="admin-2", note="撤销误操作")

            self.assertEqual(reverted["review_status"], "pending")
            self.assertIsNone(manager.reviewed.get_entry(sample_id))
            self.assertTrue(manager.delete_entry(folder))

    def test_tampered_second_review_requires_a_field_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            self._seed_entry(manager, "wrong")
            folder = manager.list_entries("wrong")[0]["folder_name"]

            with self.assertRaises(ReviewRegionRequired):
                manager.review_entry(folder, label=1, reviewer="admin-1")

    def test_review_filter_separates_pending_and_reviewed_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            self._seed_entry(manager, "correct")
            first_folder = manager.list_entries("correct")[0]["folder_name"]
            self._seed_entry(manager, "wrong")
            manager.review_entry(first_folder, label=0, reviewer="admin")

            self.assertEqual(len(manager.list_entries(review_filter="pending")), 1)
            self.assertEqual(len(manager.list_entries(review_filter="reviewed")), 1)


if __name__ == "__main__":
    unittest.main()

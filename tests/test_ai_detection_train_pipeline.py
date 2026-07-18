import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.ai_detection.train_pipeline_v2 import TrainPipeline


class TrainPipelineSourceTests(unittest.TestCase):
    def _pipeline(self, root: Path) -> TrainPipeline:
        config_path = root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "dataset": {"image_dir": str(root / "images")},
                    "feedback": {"storage_dir": str(root / "feedback")},
                    "training": {"output_dir": str(root / "models")},
                }
            ),
            encoding="utf-8",
        )
        return TrainPipeline(str(config_path))

    def test_initial_feedback_is_never_used_for_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._pipeline(root)
            pending = root / "feedback" / "wrong" / "entry"
            pending.mkdir(parents=True)
            (pending / "original.jpg").write_bytes(b"pending")
            (pending / "metadata.json").write_text(
                json.dumps({"true_label": 1}), encoding="utf-8"
            )

            self.assertEqual(pipeline._load_reviewed_dataset(None), [])

    def test_reviewed_dataset_and_grouped_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._pipeline(root)
            reviewed = root / "feedback" / "reviewed" / "normal"
            reviewed.mkdir(parents=True)
            image = reviewed / "sample__12345678.png"
            image.write_bytes(b"reviewed")
            (reviewed / "sample__12345678.json").write_text(
                json.dumps(
                    {
                        "sample_id": "a" * 64,
                        "label": 0,
                        "storage_filename": image.name,
                    }
                ),
                encoding="utf-8",
            )
            rows = pipeline._load_reviewed_dataset(None)
            self.assertEqual(rows[0][1:], (0, "reviewed:" + "a" * 64))

            images = root / "images"
            images.mkdir()
            for name in (
                "no (1).jpg", "no (1)_enhanced.jpg", "no (2).jpg", "no (3).jpg",
                "p (1).jpg", "p (1)_enhanced.jpg", "p (2).jpg", "p (3).jpg",
            ):
                (images / name).write_bytes(b"x")
            base_rows = pipeline._load_original_dataset()
            by_name = {Path(path).name: group for path, _label, group in base_rows}
            self.assertEqual(by_name["no (1).jpg"], by_name["no (1)_enhanced.jpg"])
            train_groups, validation_groups = pipeline._deterministic_group_split(
                base_rows, validation_ratio=0.34
            )
            self.assertTrue(train_groups.isdisjoint(validation_groups))
            labels = {
                label for _path, label, group in base_rows if group in validation_groups
            }
            self.assertEqual(labels, {0, 1})

    def test_manifest_uses_explicit_validation_and_test_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._pipeline(root)
            images = root / "images"
            (images / "normal").mkdir(parents=True)
            (images / "tampered").mkdir(parents=True)
            (root / "pptest").mkdir()
            for relative in (
                "normal/real.jpg",
                "normal/real_val.jpg",
                "tampered/fake.jpg",
                "tampered/fake_val.jpg",
                "tampered/replay.png",
            ):
                path = images / relative
                path.write_bytes(b"image")
            manifest = {
                "entries": [
                    {"path": "normal/real.jpg", "label": 0, "split": "train", "group_id": "n-train", "is_derived": False},
                    {"path": "normal/real_val.jpg", "label": 0, "split": "validation", "group_id": "n-val", "is_derived": False},
                    {"path": "tampered/fake.jpg", "label": 1, "split": "train", "group_id": "t-train", "is_derived": False},
                    {"path": "tampered/fake_val.jpg", "label": 1, "split": "validation", "group_id": "t-val", "is_derived": False},
                    {"path": "tampered/replay.png", "label": 1, "split": "train", "group_id": "pptest:fixed", "is_derived": False, "training_replay_regression": True},
                ]
            }
            (images / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            rows = pipeline._load_original_dataset()

            self.assertEqual(
                {Path(path).name for path, _label, _group in rows},
                {"real.jpg", "real_val.jpg", "fake.jpg", "fake_val.jpg", "replay.png"},
            )
            self.assertEqual(len(pipeline._test_samples), 0)
            self.assertEqual(len(pipeline._training_replay_samples), 1)
            self.assertEqual(pipeline._dataset_split_by_path[str((images / "normal/real_val.jpg").resolve())], "validation")

    def test_reference_index_contains_only_train_normal_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._pipeline(root)
            images = root / "images"
            (images / "normal").mkdir(parents=True)
            (images / "tampered").mkdir(parents=True)
            normal = images / "normal" / "source.png"
            production = images / "tampered" / "production.png"
            test_normal = images / "normal" / "test.png"
            image = np.full((40, 80, 3), 240, dtype=np.uint8)
            cv2.imencode(".png", image)[1].tofile(normal)
            cv2.imencode(".png", image)[1].tofile(production)
            cv2.imencode(".png", image)[1].tofile(test_normal)
            normal_hash = pipeline._sha256(normal)
            manifest = {
                "entries": [
                    {"path": "normal/source.png", "label": 0, "split": "train", "is_derived": False, "sha256": normal_hash},
                    {"path": "normal/test.png", "label": 0, "split": "test", "is_derived": False, "sha256": pipeline._sha256(test_normal)},
                    {"path": "tampered/production.png", "label": 1, "split": "train", "is_derived": False, "source": "production_tampered", "sha256": pipeline._sha256(production)},
                ]
            }
            pipeline.dataset_manifest = manifest
            output = root / "models" / "candidate"
            output.mkdir(parents=True)

            index_path = pipeline._write_normal_reference_index(output)

            self.assertIsNotNone(index_path)
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual([item["path"] for item in index["references"]], ["normal/source.png"])
            self.assertNotIn("verified_tampered_references", index)


if __name__ == "__main__":
    unittest.main()

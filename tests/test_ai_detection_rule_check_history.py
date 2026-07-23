import unittest

from app.ai_detection.services.rule_check_history import (
    MODE_RULE_CHECKS,
    MODE_RULE_PIXEL_OVERLAP,
    MODE_RULE_TIMESTAMP,
    build_pixel_overlap_outcome,
    build_rule_check_failed_outcome,
    build_rule_checks_outcome,
    build_timestamp_outcome,
)


class RuleCheckHistoryOutcomeTests(unittest.TestCase):
    def test_build_rule_checks_outcome_summary(self):
        data = {
            "pixel_overlap": {
                "pixel_overlap_score": 0.64,
                "alert": True,
                "hard_tamper": False,
            },
            "timestamp": {
                "risk": 0.58,
                "hard_tamper": False,
                "business_mismatch": True,
                "anomalies": ["business_visible_datetime_mismatch"],
            },
            "semantic": {
                "risk": 0.45,
                "hard_tamper": False,
                "anomalies": ["uppercase_lowercase_amount_mismatch"],
            },
            "hard_tamper_flags": {"pixel_overlap": False, "timestamp": False, "semantic": False},
            "reason": "测试",
        }
        outcome = build_rule_checks_outcome(
            data,
            bbox=[120, 80, 400, 140],
            document_time="2026-05-28 11:32:00",
        )
        self.assertEqual(outcome["check_type"], MODE_RULE_CHECKS)
        self.assertTrue(outcome["summary"]["pixel_alert"])
        self.assertEqual(outcome["summary"]["timestamp_risk"], 0.58)
        self.assertTrue(outcome["summary"]["business_mismatch"])
        self.assertEqual(outcome["summary"]["semantic_anomaly_codes"], ["uppercase_lowercase_amount_mismatch"])
        self.assertFalse(outcome["summary"]["any_hard_tamper"])

    def test_build_pixel_overlap_outcome(self):
        data = {
            "pixel_overlap_score": 0.94,
            "alert": True,
            "hard_tamper": True,
        }
        outcome = build_pixel_overlap_outcome(data, bbox=[1, 2, 3, 4])
        self.assertEqual(outcome["check_type"], MODE_RULE_PIXEL_OVERLAP)
        self.assertTrue(outcome["summary"]["any_hard_tamper"])

    def test_build_timestamp_outcome(self):
        data = {
            "risk": 0.38,
            "hard_tamper": False,
            "business_mismatch": False,
            "anomalies": ["transaction_time_unparsed"],
        }
        outcome = build_timestamp_outcome(data, document_time="2026-01-01 12:00:00")
        self.assertEqual(outcome["check_type"], MODE_RULE_TIMESTAMP)
        self.assertEqual(outcome["summary"]["anomaly_codes"], ["transaction_time_unparsed"])

    def test_build_rule_checks_outcome_keeps_watermark_as_hard_tamper(self):
        outcome = build_rule_checks_outcome(
            {
                "pixel_overlap": None,
                "timestamp": {"risk": 0.0, "hard_tamper": False, "anomalies": []},
                "semantic": {"risk": 0.0, "hard_tamper": False, "anomalies": []},
                "ai_watermark": {
                    "hard_tamper": True,
                    "matched_text": "豆包AI生成",
                    "evidence_type": "ai_generated_document",
                },
                "hard_tamper_flags": {"doubao_ai_watermark": True},
                "reason": "识别到“豆包AI生成”水印，按业务规则直接判定篡改",
            }
        )

        self.assertTrue(outcome["summary"]["any_hard_tamper"])
        self.assertTrue(outcome["hard_tamper_flags"]["doubao_ai_watermark"])
        self.assertEqual(outcome["ai_watermark"]["matched_text"], "豆包AI生成")

    def test_build_failed_outcome(self):
        outcome = build_rule_check_failed_outcome(
            MODE_RULE_PIXEL_OVERLAP,
            "无法读取图片",
            bbox=[10, 20, 30, 40],
        )
        self.assertEqual(outcome["error_msg"], "无法读取图片")
        self.assertEqual(outcome["request"]["bbox"], [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()

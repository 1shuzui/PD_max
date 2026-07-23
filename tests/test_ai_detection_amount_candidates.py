import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.ai_detection.core.amount_candidates import (
    AmountCandidate,
    OCRToken,
    build_amount_candidates,
    detect_certificate_document_override,
    find_uppercase_amount_pairs,
    is_account_context_token,
    parse_chinese_uppercase_amount,
    parse_decimal_amount,
    tokenize_ocr_results,
)


class AmountCandidateTests(unittest.TestCase):
    def test_account_value_split_into_multiple_tokens_is_not_an_amount_candidate(self):
        tokens = [
            OCRToken("收款方账号", "收款方账号", (173, 1085, 383, 1137), 0.9, 210, 52, 1111.0),
            OCRToken("6230", "6230", (647, 1088, 757, 1135), 0.9, 110, 47, 1111.5),
            OCRToken("***3 076", "***3076", (931, 1088, 1104, 1135), 0.9, 173, 47, 1111.5),
            OCRToken("转账金额", "转账金额", (171, 1238, 343, 1289), 0.9, 172, 51, 1263.5),
            OCRToken("339,041.00元", "339,041.00元", (843, 1238, 1102, 1289), 0.9, 259, 51, 1263.5),
        ]

        self.assertTrue(is_account_context_token(tokens[1], tokens))
        candidates = build_amount_candidates(tokens, (1280, 2781, 3))
        candidate_bboxes = [item.bbox for item in candidates]
        self.assertNotIn((647, 1088, 757, 1135), candidate_bboxes)
        self.assertIn((843, 1238, 1102, 1289), candidate_bboxes)

    def test_uppercase_lowercase_amount_pair_parses_only_reliable_text(self):
        tokens = [
            OCRToken("小写:12,345.60元", "小写:12,345.60元", (100, 200, 300, 230), 0.9, 200, 30, 215.0),
            OCRToken("大写:壹万贰仟叁佰肆拾伍元陆角", "大写:壹万贰仟叁佰肆拾伍元陆角", (100, 240, 440, 270), 0.9, 340, 30, 255.0),
        ]

        self.assertEqual(parse_decimal_amount("小写:12,345.60元"), Decimal("12345.60"))
        self.assertEqual(parse_chinese_uppercase_amount("大写:壹万贰仟叁佰肆拾伍元陆角"), Decimal("12345.6"))
        pairs = find_uppercase_amount_pairs(tokens)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["comparison_available"])
        self.assertTrue(pairs[0]["consistent"])

    def test_uppercase_amount_with_ocr_noise_is_not_parsed(self):
        tokens = [
            OCRToken("小:51510,.00元", "小:51510,.00元", (141, 220, 232, 234), 0.38, 91, 14, 227.0),
            OCRToken("2另全8", "2另全8", (72, 228, 116, 241), 0.11, 44, 13, 234.5),
            OCRToken("大丐:侨4佰~[〉", "大丐:侨4佰~[〉", (141, 235, 251, 249), 0.01, 110, 14, 242.0),
        ]

        pairs = find_uppercase_amount_pairs(tokens)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["small_amount"], "51510.00")
        self.assertFalse(pairs[0]["comparison_available"])
        self.assertIsNone(pairs[0]["consistent"])
        self.assertEqual(pairs[0]["reason"], "uppercase_ocr_unreadable")

    def test_prefers_amount_region_over_time_and_order_number(self):
        ocr_results = [
            (
                [[10, 10], [90, 10], [90, 40], [10, 40]],
                "11:32",
                0.99,
            ),
            (
                [[100, 220], [620, 220], [620, 270], [100, 270]],
                "转账金额 ¥1,234.56",
                0.98,
            ),
            (
                [[100, 360], [620, 360], [620, 410], [100, 410]],
                "订单号 123456789012",
                0.97,
            ),
        ]

        candidates = build_amount_candidates(
            tokenize_ocr_results(ocr_results),
            (1000, 800, 3),
        )

        self.assertGreater(len(candidates), 0)
        top = candidates[0]
        self.assertEqual(top.bbox, (100, 220, 620, 270))
        self.assertIn("金额", top.text)
        self.assertIn("money_regex", top.match_flags)
        self.assertNotEqual(top.clean_text, "11:32")

    @patch("app.ai_detection.core.amount_candidates.OriginalityChecker.extract_features")
    def test_detects_certificate_amount_row_override(self, mock_extract_features):
        mock_extract_features.return_value = (
            {
                "has_exif": 0,
                "size_per_pixel": 0.24,
                "color_entropy": 1.58,
            },
            False,
            "",
        )
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        tokens = [
            OCRToken(
                text="欲信支付转然电孑凭证",
                clean_text="欲信支付转然电孑凭证",
                bbox=(80, 40, 320, 72),
                conf=0.12,
                width=240,
                height=32,
                center_y=56.0,
            )
        ]
        candidates = [
            AmountCandidate(
                source="token",
                text="4555890元",
                clean_text="4555890元",
                bbox=(210, 250, 360, 280),
                ocr_confidence=0.18,
                amount_score=0.4,
                match_flags="currency_hint|compact_digits",
            )
        ]

        class _StubReader:
            def readtext(self, *_args, **_kwargs):
                return [
                    ([[0, 0], [10, 0], [10, 10], [0, 10]], "收款方", 0.9),
                    ([[0, 0], [10, 0], [10, 10], [0, 10]], "小:5558900元", 0.9),
                    ([[0, 0], [10, 0], [10, 10], [0, 10]], "交易金额", 0.9),
                    ([[0, 0], [10, 0], [10, 10], [0, 10]], "大写:伍万伍仟伍佰捌拾玖圆整", 0.9),
                ]

        override = detect_certificate_document_override(
            image_path=Path("/tmp/mock-certificate.png"),
            image=image,
            tokens=tokens,
            candidates=candidates,
            ocr_reader=_StubReader(),
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertEqual(override["result"], "篡改")
        self.assertEqual(override["reason"], "电子凭证金额行结构异常")
        self.assertEqual(override["bbox_xyxy"], [0, 190, 585, 340])

    @patch("app.ai_detection.core.amount_candidates.OriginalityChecker.extract_features")
    def test_detects_certificate_screen_photo_override(self, mock_extract_features):
        mock_extract_features.return_value = (
            {
                "has_exif": 0,
                "size_per_pixel": 2.46,
                "color_entropy": 7.54,
                "noise_mean": 295.17,
                "noise_std": 354.66,
            },
            False,
            "",
        )
        image = np.zeros((2863, 3000, 3), dtype=np.uint8)
        tokens = [
            OCRToken(
                text="微信支付转账电子凭证",
                clean_text="微信支付转账电子凭证",
                bbox=(1100, 338, 1822, 436),
                conf=0.6577,
                width=722,
                height=98,
                center_y=387.0,
            ),
            OCRToken(
                text="申咂蒹{206031617.162",
                clean_text="申咂蒹{206031617.162",
                bbox=(243, 497, 904, 569),
                conf=0.0014,
                width=661,
                height=72,
                center_y=533.0,
            ),
            OCRToken(
                text=".=6900",
                clean_text=".=6900",
                bbox=(746, 991, 1142, 1058),
                conf=0.0039,
                width=396,
                height=67,
                center_y=1024.5,
            ),
            OCRToken(
                text="辜{",
                clean_text="辜{",
                bbox=(388, 1040, 593, 1109),
                conf=0.0,
                width=205,
                height=69,
                center_y=1074.5,
            ),
            OCRToken(
                text="夫-仟叁b拾玖獾",
                clean_text="夫-仟叁b拾玖獾",
                bbox=(748, 1059, 1393, 1133),
                conf=0.0,
                width=645,
                height=74,
                center_y=1096.0,
            ),
        ]
        tokens.extend(
            OCRToken(
                text=f"噪声{i}",
                clean_text=f"噪声{i}",
                bbox=(100 + i * 10, 1500 + i * 12, 220 + i * 10, 1560 + i * 12),
                conf=0.0,
                width=120,
                height=60,
                center_y=1530.0 + i * 12,
            )
            for i in range(12)
        )
        candidates = [
            AmountCandidate(
                source="token",
                text="申咂蒹{206031617.162",
                clean_text="申咂蒹{206031617.162",
                bbox=(243, 497, 904, 569),
                ocr_confidence=0.0014,
                amount_score=1.35,
                match_flags="money_regex|prominent",
            )
        ]

        class _UnusedReader:
            def readtext(self, *_args, **_kwargs):
                raise AssertionError("screen photo override should not read certificate rows")

        override = detect_certificate_document_override(
            image_path=Path("/tmp/mock-certificate-screen-photo.png"),
            image=image,
            tokens=tokens,
            candidates=candidates,
            ocr_reader=_UnusedReader(),
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertEqual(override["result"], "篡改")
        self.assertEqual(override["reason"], "电子凭证翻拍纹理明显且金额区OCR异常")
        self.assertEqual(override["bbox_xyxy"], [388, 991, 1393, 1133])


if __name__ == "__main__":
    unittest.main()

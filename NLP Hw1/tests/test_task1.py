"""检查词袋语义、数据划分和评价指标的关键性质。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1 import build_vocabulary, evaluate, get_split, tokenize, vectorize


class Task1Tests(unittest.TestCase):
    """验证作业要求涉及的关键计算逻辑。"""

    def test_binary_and_frequency(self):
        """用手算例子确认二值表示与原始计数的差异。"""
        vocabulary = build_vocabulary(["apple apple banana", "banana carrot"])
        texts = ["apple apple banana", "carrot carrot carrot", "unknown", ""]
        frequency = vectorize(texts, vocabulary)
        binary = vectorize(texts, vocabulary, binary=True)
        np.testing.assert_array_equal(frequency.toarray(), [[2, 1, 0], [0, 0, 3], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(binary.toarray(), [[1, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(frequency.format, "csr")

    def test_vocabulary_excludes_unseen_words(self):
        """确保测试集的新词不会进入训练词表。"""
        vocabulary = build_vocabulary(["Train apple"])
        vectorize(["testonly apple"], vocabulary)
        self.assertNotIn("testonly", vocabulary)
        self.assertEqual(tokenize("A, APPLE 123!"), ["a", "apple", "123"])

    def test_split_reuse_and_integrity(self):
        """检查划分完整互斥、固定可复用且能识别数据变化。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            splits = get_split(path, 11519, "数据指纹")
            self.assertEqual([len(value) for value in splits.values()], [9215, 1152, 1152])
            combined = np.concatenate(list(splits.values()))
            np.testing.assert_array_equal(np.sort(combined), np.arange(11519))
            again = get_split(path, 11519, "数据指纹")
            for name in splits:
                np.testing.assert_array_equal(splits[name], again[name])
            with self.assertRaises(ValueError):
                get_split(path, 11519, "变化后的指纹")
            stored = json.loads(path.read_text())
            stored["test"][0] = stored["train"][0]
            path.write_text(json.dumps(stored))
            with self.assertRaises(ValueError):
                get_split(path, 11519, "数据指纹")

    def test_macro_f1_is_not_accuracy(self):
        """用类别不平衡例子验证宏平均指标的计算。"""
        metrics = evaluate(["a", "a", "a", "b"], ["a", "a", "a", "a"], ["a", "b"])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], 3 / 7)


if __name__ == "__main__":
    unittest.main()

"""使用两种词袋表示和逻辑回归完成新闻分类实验。"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import warnings

import numpy as np
import scipy
from scipy.sparse import csr_matrix
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]  # 从代码目录定位作业目录，不依赖执行位置。
SEED = 42  # 固定随机种子，供本次与后续实验复用。
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")  # 保留单字符词和数字，忽略标点。


def tokenize(text):
    """将文本转为小写并按词边界分词。"""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts):
    """仅根据训练文本构建按字典序排列的词表。"""
    words = {word for text in texts for word in tokenize(text)}
    return {word: index for index, word in enumerate(sorted(words))}


def vectorize(texts, vocabulary, binary=False):
    """把文本转换为二值词袋或原始词频的稀疏矩阵。"""
    indices, values, indptr = [], [], [0]  # 分别保存列索引、非零值和每行起点。
    for text in texts:
        counts = Counter(tokenize(text))  # 统计当前文档中每个词的出现次数。
        entries = sorted(
            (vocabulary[word], count)
            for word, count in counts.items() if word in vocabulary
        )
        for column, count in entries:
            indices.append(column)
            values.append(1 if binary else count)  # 二值表示只记录是否出现。
        indptr.append(len(indices))
    return csr_matrix(
        (values, indices, indptr),
        shape=(len(texts), len(vocabulary)),
        dtype=np.float64,  # 使用逻辑回归支持的双精度稀疏输入。
    )


def load_data(path):
    """读取新闻数据并检查列名、空文本和空标签。"""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "label"]:
            raise ValueError("数据必须包含且仅包含 text、label 两列。")
        rows = list(reader)
    if not rows or any(
        set(row) != {"text", "label"} or not row["text"]
        or not row["text"].strip() or not row["label"] or not row["label"].strip()
        for row in rows
    ):
        raise ValueError("数据包含空文本、空标签或格式错误的行。")
    texts = np.asarray([row["text"] for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    return texts, labels


def get_split(path, sample_count, data_hash):
    """保存或校验固定的数据划分，防止后续任务改变测试集。"""
    permutation = np.random.RandomState(SEED).permutation(sample_count)
    train_end = int(sample_count * 0.8)  # 训练集取前百分之八十，向下取整。
    validation_end = int(sample_count * 0.9)  # 剩余样本依次分给验证集和测试集。
    expected = {
        "seed": SEED,
        "data_sha256": data_hash,
        "sample_count": sample_count,
        "index_description": "去除表头后的零起始数据行索引",
        "train": permutation[:train_end].tolist(),
        "validation": permutation[train_end:validation_end].tolist(),
        "test": permutation[validation_end:].tolist(),
    }
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != expected:
            raise ValueError("已有划分与当前数据或划分规则不一致，请检查数据和索引文件。")
    else:
        save_json(path, expected)
    return {name: np.asarray(expected[name]) for name in ("train", "validation", "test")}


def save_json(path, value):
    """将结构化实验记录保存为可读的中文兼容文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(labels, predictions, classes):
    """计算准确率、宏平均分数及每个类别的详细指标。"""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "classification_report": classification_report(
            labels, predictions, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }


def main():
    """运行两组词袋分类实验并保存可复现的评估记录。"""
    parser = argparse.ArgumentParser(description="运行作业一的两种词袋分类实验。")
    parser.add_argument("--data", type=Path, default=ROOT / "data/nyt.csv", help="新闻数据路径")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="结构化结果目录")
    args = parser.parse_args()
    texts, labels = load_data(args.data)
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()  # 识别原始数据是否变化。
    splits = get_split(args.output / "nyt_split.json", len(texts), data_hash)
    classes = sorted(set(labels))  # 固定指标和混淆矩阵的类别顺序。
    if set(labels[splits["train"]]) != set(classes):
        raise ValueError("训练集没有覆盖全部类别，无法完成当前分类实验。")
    vocabulary = build_vocabulary(texts[splits["train"]])
    save_json(args.output / "task1_vocabulary.json", vocabulary)
    parameters = dict(
        C=1.0,  # 正则化强度的倒数，数值越小约束越强。
        solver="lbfgs",  # 使用适合多项逻辑回归的优化器。
        max_iter=2000,  # 优化迭代次数上限。
        tol=1e-4,  # 收敛判定容差。
        penalty="l2",  # 使用二范数正则化。
        fit_intercept=True,  # 训练截距项。
        class_weight=None,  # 不额外调整类别权重。
        random_state=SEED,  # 保持统一的随机种子设置。
    )
    results = {
        "environment": f"Python {sys.version.split()[0]} / {platform.platform()} / NumPy {np.__version__} / SciPy {scipy.__version__} / scikit-learn {sklearn.__version__}",
        "sample_count": len(texts), "data_sha256": data_hash, "classes": classes,
        "vocabulary_size": len(vocabulary), "parameters": parameters,
        "split_counts": {name: dict(Counter(labels[index])) for name, index in splits.items()},
        "methods": {},
    }
    test_predictions = {}  # 保留两种方法在同一测试集上的逐条预测。
    for name, binary in (("Binary Bag of Words", True), ("Word Frequency", False)):
        print(f"正在运行 {name}，训练词表大小：{len(vocabulary)}", flush=True)
        matrices = {part: vectorize(texts[index], vocabulary, binary) for part, index in splits.items()}
        model = LogisticRegression(**parameters)
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(matrices["train"], labels[splits["train"]])
        method_result = {"iterations": int(model.n_iter_.max())}
        for part in ("validation", "test"):
            predictions = model.predict(matrices[part])
            method_result[part] = evaluate(labels[splits[part]], predictions, classes)
            if part == "test":
                test_predictions[name] = predictions
        results["methods"][name] = method_result
        print(f"测试 Accuracy={method_result['test']['accuracy']:.6f}，Macro-F1={method_result['test']['macro_f1']:.6f}，迭代次数={method_result['iterations']}", flush=True)
    save_json(args.output / "task1_metrics.json", results)
    with (args.output / "task1_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")  # 统一换行格式，便于版本管理。
        writer.writerow(["row_index", "true_label", "binary_prediction", "frequency_prediction"])
        for position, index in enumerate(splits["test"]):
            writer.writerow([int(index), labels[index],
                             test_predictions["Binary Bag of Words"][position],
                             test_predictions["Word Frequency"][position]])
    print(f"实验完成，指标、预测和词表已保存：{args.output}", flush=True)


if __name__ == "__main__":
    main()

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


ROOT = Path(__file__).resolve().parent  # 作业目录，不依赖命令执行位置。
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


def write_report(path, results):
    """将真实运行结果、实验设置和方法比较写入中文报告。"""
    lines = [
        "# Task 1 词袋模型实验结果", "",
        "## 实验目标与环境", "",
        "使用 Binary Bag of Words 与 Word Frequency 表示 NYT 新闻，统一训练 Logistic Regression。",
        "Word Frequency 表示原始出现次数，不做长度归一化或 TF-IDF 加权。", "",
        f"运行环境：{results['environment']}。", "",
        "## 数据与划分", "",
        f"NYT 共 {results['sample_count']} 条样本；类别为 {', '.join(results['classes'])}。AG News 未参与本实验。",
        "固定种子 42，使用 NumPy RandomState.permutation 随机打乱，按 80% / 10% / 10% 顺序切分，不额外分层。",
        "训练结束位置为 floor(0.8N)，验证结束位置为 floor(0.9N)，余下样本用于测试。",
        "行索引固定保存在 `../results/nyt_split.json`，后续任务应复用该文件。", "",
        "| 数据集 | 样本数 | business | politics | sports |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, counts in results["split_counts"].items():
        lines.append(f"| {name} | {sum(counts.values())} | {counts.get('business', 0)} | {counts.get('politics', 0)} | {counts.get('sports', 0)} |")
    lines += [
        "", f"NYT 文件 SHA-256：`{results['data_sha256']}`。", "",
        "## 文本表示与训练设置", "",
        "文本转小写，以 Unicode 正则 `(?u)\\b\\w+\\b` 提取词项；保留单字符词和数字，去掉标点，不移除停用词、不做词干化。",
        "作业中的 NLTK 分词是可选建议，本实验使用不需要下载模型的确定性正则分词。",
        f"词表仅从训练集构建，共 {results['vocabulary_size']} 个词项；验证集和测试集中的未知词忽略。",
        "两种表示均自行计数并构建 CSR 稀疏矩阵，Binary 取 0/1，Frequency 保留原始计数。",
        "两种方法共用词表、划分和分类器参数：L2 正则化，C=1.0，solver=lbfgs，max_iter=2000，tol=1e-4，fit_intercept=True，class_weight=None。",
        "三分类使用多项逻辑回归；固定 random_state=42，并限制数值计算为单线程。",
        "未进行超参数搜索。验证集仅用于辅助报告，不参与词表构建或训练；测试集仅用于最终评价。",
        "收敛警告作为错误处理，不记录未收敛训练结果。", "",
        "## 评价结果", "",
        "Accuracy 为预测正确的样本比例；Macro-F1 为三个类别各自 F1 的算术平均，指标越大越好。", "",
        "| 表示方法 | 验证 Accuracy | 验证 Macro-F1 | 测试 Accuracy | 测试 Macro-F1 | 迭代次数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results["methods"].items():
        val, test = result["validation"], result["test"]
        lines.append(f"| {name} | {val['accuracy']:.6f} | {val['macro_f1']:.6f} | {test['accuracy']:.6f} | {test['macro_f1']:.6f} | {result['iterations']} |")
    for name, result in results["methods"].items():
        lines += ["", f"### {name} 测试集分类明细", "",
                  "| 类别 | Precision | Recall | F1 | 样本数 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for label in results["classes"]:
            item = result["test"]["classification_report"][label]
            lines.append(f"| {label} | {item['precision']:.6f} | {item['recall']:.6f} | {item['f1-score']:.6f} | {int(item['support'])} |")
        lines += ["", "混淆矩阵（行是真实类别，列是预测类别；顺序为 business、politics、sports）：", "", "```text"]
        lines.extend(str(row) for row in result["test"]["confusion_matrix"])
        lines.append("```")
    binary = results["methods"]["Binary Bag of Words"]["test"]
    frequency = results["methods"]["Word Frequency"]["test"]
    lines += [
        "", "## 结果分析", "",
        f"Binary 相对 Frequency 的测试 Accuracy 差值为 {(binary['accuracy'] - frequency['accuracy']) * 100:+.4f} 个百分点，Macro-F1 差值为 {(binary['macro_f1'] - frequency['macro_f1']) * 100:+.4f} 个百分点。",
        "二值表示削弱同一新闻中重复用词的影响；原始词频保留重复次数，也会受到文档长度的影响。上述差异来自本次固定划分，不能据此断言某种表示在所有数据上都更好。",
        "数据中的 sports 类占比较高，Accuracy 容易受到多数类表现影响，因此应同时观察 Macro-F1 和各类别明细。", "",
        "## 复现与输出", "",
        "在仓库根目录完成依赖安装后执行：", "", "```bash",
        '.venv/bin/python "NLP Hw1/task1.py"',
        '.venv/bin/python -m unittest discover -s "NLP Hw1/tests" -v', "```", "",
        "- `../results/task1_metrics.json`：完整指标、环境、超参数、收敛迭代次数。",
        "- `../results/task1_predictions.csv`：测试样本原始行索引、真实标签、两种方法的预测。",
        "- `../results/nyt_split.json`：固定划分与原始数据指纹。",
        "- `../results/task1_vocabulary.json`：训练集词项到列索引的映射。", "",
        "本报告由实验脚本根据实际运行结果生成。当前阶段按要求提交 Markdown 记录；完整作业最终要求的 PDF 报告留待后续任务完成后统一整理。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """运行两组词袋分类实验并保存可复现的评估记录。"""
    parser = argparse.ArgumentParser(description="运行作业一的两种词袋分类实验。")
    parser.add_argument("--data", type=Path, default=ROOT / "data/nyt.csv", help="新闻数据路径")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="结构化结果目录")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/Task1实验结果.md", help="中文报告路径")
    args = parser.parse_args()
    texts, labels = load_data(args.data)
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()  # 识别原始数据是否变化。
    splits = get_split(args.output / "nyt_split.json", len(texts), data_hash)
    classes = sorted(set(labels))  # 固定指标和混淆矩阵的类别顺序。
    if set(labels[splits["train"]]) != set(classes):
        raise ValueError("训练集没有覆盖全部类别，无法完成当前分类实验。")
    vocabulary = build_vocabulary(texts[splits["train"]])
    save_json(args.output / "task1_vocabulary.json", vocabulary)
    parameters = dict(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-4,
                      penalty="l2", fit_intercept=True, class_weight=None, random_state=SEED)
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
    write_report(args.report, results)
    print(f"实验完成，报告已保存：{args.report}", flush=True)


if __name__ == "__main__":
    main()

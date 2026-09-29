"""使用三组百维词向量的平均表示完成新闻分类实验。"""

import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import sys
import warnings

import gensim
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from task1 import ROOT, SEED, evaluate, get_split, load_data, save_json, tokenize


DIMENSION = 100  # 作业要求的词向量维数。
METHODS = ("GloVe", "Word2Vec_AG", "Word2Vec_NYT")  # 固定实验与输出顺序。
W2V_PARAMETERS = {
    "vector_size": DIMENSION,  # 所有词向量均为一百维。
    "window": 5,  # 上下文窗口的最大半径。
    "min_count": 2,  # 忽略训练语料中只出现一次的词。
    "sg": 1,  # 使用根据中心词预测上下文的训练方式。
    "negative": 5,  # 每个正样本对应的负采样数量。
    "hs": 0,  # 关闭层次化输出层，使用负采样目标。
    "sample": 1e-3,  # 对高频词进行下采样。
    "epochs": 10,  # 两份语料使用相同训练轮数。
    "alpha": 0.025,  # 初始学习率。
    "min_alpha": 0.0001,  # 最终学习率。
    "workers": 1,  # 使用单线程以避免训练调度引入随机差异。
    "seed": SEED,  # 与数据划分统一的随机种子。
    "sorted_vocab": 1,  # 按词频排序词表，保持初始化顺序稳定。
    "shrink_windows": True,  # 在最大窗口内随机选取实际窗口。
    "batch_words": 10000,  # 每批训练的目标词数。
    "ns_exponent": 0.75,  # 负采样分布的词频指数。
}
LR_PARAMETERS = {
    "C": 1.0,  # 正则化强度的倒数，与第一部分保持一致。
    "solver": "lbfgs",  # 多项逻辑回归的优化器。
    "max_iter": 2000,  # 优化器迭代次数上限。
    "tol": 1e-4,  # 收敛判断的容差。
    "penalty": "l2",  # 使用二范数正则化。
    "fit_intercept": True,  # 拟合截距项。
    "class_weight": None,  # 不额外修改类别权重。
    "random_state": SEED,  # 固定随机种子。
}


class TrainingProgress(CallbackAny2Vec):
    """记录完整语料训练的轮次进度。"""

    def __init__(self, enabled):
        """初始化进度计数并控制小规模测试是否输出。"""
        self.epoch = 0
        self.enabled = enabled

    def on_epoch_end(self, model):
        """每轮训练完成后输出当前轮次。"""
        self.epoch += 1
        if self.enabled:
            print(f"Word2Vec 已完成 {self.epoch}/{model.epochs} 轮", flush=True)


def file_sha256(path):
    """分块计算文件指纹，避免一次性加载大型词向量文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    """提供不依赖解释器随机哈希种子的稳定初始化哈希。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def read_ag(path):
    """读取并检查仅含文本列的新闻词向量训练语料。"""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text"]:
            raise ValueError("AG News 必须只包含 text 列。")
        documents = []
        for row in reader:
            if set(row) != {"text"} or not row["text"] or not row["text"].strip():
                raise ValueError("AG News 含有空文本或格式错误的记录。")
            documents.append(tokenize(row["text"]))
    if not documents:
        raise ValueError("AG News 语料为空。")
    return documents


def train_word2vec(documents):
    """仅用传入语料训练百维词向量并返回训练统计。"""
    if not documents or not any(documents):
        raise ValueError("词向量训练语料不能为空。")
    with threadpool_limits(limits=1):
        model = Word2Vec(sentences=documents, hashfxn=stable_hash,
                         callbacks=[TrainingProgress(len(documents) >= 1000)], **W2V_PARAMETERS)
    if not np.isfinite(model.wv.vectors).all():
        raise ValueError("训练得到的词向量包含非有限数值。")
    statistics = {
        "documents": len(documents),
        "tokens": sum(map(len, documents)),
        "vocabulary_size": len(model.wv),
        "epochs": model.epochs,
        "vector_sha256": hashlib.sha256(model.wv.vectors.tobytes()).hexdigest(),
        "vocabulary_sha256": hashlib.sha256("\n".join(model.wv.index_to_key).encode()).hexdigest(),
    }
    return model.wv, statistics


def load_glove(path, needed_words):
    """读取官方百维向量，仅保留待表示文本需要的词以节省内存。"""
    vectors = {}
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip().split()
            if len(fields) != DIMENSION + 1:
                raise ValueError(f"GloVe 第 {line_number} 行不是百维向量。")
            rows += 1
            word = fields[0]
            if word in needed_words:
                vector = np.asarray(fields[1:], dtype=np.float32)
                if not np.isfinite(vector).all():
                    raise ValueError(f"GloVe 第 {line_number} 行包含非有限数值。")
                if word in vectors:
                    raise ValueError(f"GloVe 出现重复词项：{word}")
                vectors[word] = vector
    if not vectors:
        raise ValueError("GloVe 与当前文本没有任何可用词项。")
    return vectors, {"source_vocabulary_size": rows, "loaded_vocabulary_size": len(vectors)}


def document_vectors(documents, vectors):
    """按出现次数平均有效词向量，忽略未知词并统计覆盖情况。"""
    features = np.zeros((len(documents), DIMENSION), dtype=np.float64)
    total_tokens, matched_tokens, empty_documents = 0, 0, 0
    for index, words in enumerate(documents):
        valid = [vectors[word] for word in words if word in vectors]  # 重复词仍按出现次数参与平均。
        total_tokens += len(words)
        matched_tokens += len(valid)
        if valid:
            features[index] = np.mean(valid, axis=0, dtype=np.float64)  # 分母仅计入有效词数。
        else:
            empty_documents += 1  # 没有有效词的文本保留全零向量。
    statistics = {
        "documents": len(documents), "tokens": total_tokens,
        "matched_tokens": matched_tokens,
        "coverage": matched_tokens / total_tokens if total_tokens else 0.0,
        "zero_vector_documents": empty_documents,
    }
    return features, statistics


def fit_and_evaluate(documents, labels, splits, vectors, classes):
    """仅在训练集拟合分类器并分别评价验证集和测试集。"""
    matrices, coverage = {}, {}
    for part, indices in splits.items():
        matrices[part], coverage[part] = document_vectors([documents[i] for i in indices], vectors)
        if not np.isfinite(matrices[part]).all():
            raise ValueError(f"{part} 文档向量包含非有限数值。")
    classifier = LogisticRegression(**LR_PARAMETERS)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)  # 未收敛时停止，避免静默记录无效结果。
        warnings.simplefilter("error", RuntimeWarning)  # 数值计算警告也必须处理后再记录结果。
        classifier.fit(matrices["train"], labels[splits["train"]])
    if not np.isfinite(classifier.coef_).all() or not np.isfinite(classifier.intercept_).all():
        raise ValueError("分类器参数包含非有限数值。")
    result = {"iterations": int(classifier.n_iter_.max()), "coverage": coverage}
    test_predictions = None
    for part in ("validation", "test"):
        predictions = classifier.predict(matrices[part])
        result[part] = evaluate(labels[splits[part]], predictions, classes)
        if part == "test":
            test_predictions = predictions.tolist()
    return result, test_predictions


def main():
    """执行三组词向量分类实验并保存指标和预测。"""
    parser = argparse.ArgumentParser(description="运行三组百维词向量分类实验。")
    parser.add_argument("--glove", type=Path, default=ROOT / "data/glove/glove.6B.100d.txt", help="官方百维词向量路径")
    args = parser.parse_args()
    if not args.glove.is_file():
        raise FileNotFoundError("缺少 GloVe 百维文件，请先运行 src/prepare_glove.py。")
    split_path = ROOT / "results/nyt_split.json"  # 必须复用已经固定的划分文件。
    if not split_path.is_file():
        raise FileNotFoundError("缺少 Task 1 的固定划分，不能重新定义测试集。")
    texts, labels = load_data(ROOT / "data/nyt.csv")
    nyt_hash = file_sha256(ROOT / "data/nyt.csv")
    splits = get_split(split_path, len(texts), nyt_hash)
    classes = sorted(set(labels))
    if set(labels[splits["train"]]) != set(classes):
        raise ValueError("训练集没有覆盖所有类别。")
    documents = [tokenize(text) for text in texts]
    del texts  # 分词完成后释放原始文本数组。
    results = {
        "environment": f"Python {sys.version.split()[0]} / {platform.platform()} / NumPy {np.__version__} / SciPy {version('scipy')} / Gensim {gensim.__version__} / scikit-learn {version('scikit-learn')}",
        "classes": classes, "seed": SEED, "dimension": DIMENSION,
        "word2vec_parameters": W2V_PARAMETERS, "classifier_parameters": LR_PARAMETERS,
        "initialization_hash": "SHA-256 前四字节的小端整数",
        "inputs": {"nyt_sha256": nyt_hash, "ag_sha256": file_sha256(ROOT / "data/ag.csv"),
                   "split_sha256": file_sha256(split_path), "glove_sha256": file_sha256(args.glove)},
        "split_counts": {part: dict(Counter(labels[index])) for part, index in splits.items()},
        "methods": {},
    }
    predictions = {"row_indices": splits["test"].tolist(),
                   "true_labels": labels[splits["test"]].tolist(), "methods": {}}
    for name in METHODS:
        print(f"开始 {name} 词向量准备与分类", flush=True)
        if name == "GloVe":
            needed = {word for document in documents for word in document}
            vectors, embedding = load_glove(args.glove, needed)
            if embedding["source_vocabulary_size"] != 400000:
                raise ValueError("GloVe 词表大小不符合官方 glove.6B 的四十万词，请检查文件完整性。")
        elif name == "Word2Vec_AG":
            corpus = read_ag(ROOT / "data/ag.csv")
            vectors, embedding = train_word2vec(corpus)
            del corpus
        else:
            vectors, embedding = train_word2vec([documents[i] for i in splits["train"]])
        result, predicted = fit_and_evaluate(documents, labels, splits, vectors, classes)
        result["embedding"] = embedding
        results["methods"][name] = result
        predictions["methods"][name] = predicted
        print(f"{name} 测试 Accuracy={result['test']['accuracy']:.6f}，Macro-F1={result['test']['macro_f1']:.6f}", flush=True)
        del vectors
    save_json(ROOT / "results/task2_metrics.json", results)
    save_json(ROOT / "results/task2_predictions.json", predictions)
    print("三组实验已完成，指标与逐条预测已保存。", flush=True)


if __name__ == "__main__":
    main()

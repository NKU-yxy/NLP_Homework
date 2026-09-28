# Task 1 词袋模型实验结果

## 实验目标与环境

使用 Binary Bag of Words 与 Word Frequency 表示 NYT 新闻，统一训练 Logistic Regression。
Word Frequency 表示原始出现次数，不做长度归一化或 TF-IDF 加权。

运行环境：Python 3.12.14 / macOS-26.6-arm64-arm-64bit / NumPy 2.2.6 / SciPy 1.15.3 / scikit-learn 1.6.1。

## 数据与划分

NYT 共 11519 条样本；类别为 business, politics, sports。AG News 未参与本实验。
固定种子 42，使用 NumPy RandomState.permutation 随机打乱，按 80% / 10% / 10% 顺序切分，不额外分层。
训练结束位置为 floor(0.8N)，验证结束位置为 floor(0.9N)，余下样本用于测试。
行索引固定保存在 `../results/nyt_split.json`，后续任务应复用该文件。

| 数据集 | 样本数 | business | politics | sports |
| --- | ---: | ---: | ---: | ---: |
| train | 9215 | 1157 | 1134 | 6924 |
| validation | 1152 | 124 | 152 | 876 |
| test | 1152 | 148 | 165 | 839 |

NYT 文件 SHA-256：`de12ebe7f41316b798896bf975da840e2998f82393328d375fe5456a14e9eb2d`。

## 文本表示与训练设置

文本转小写，以 Unicode 正则 `(?u)\b\w+\b` 提取词项；保留单字符词和数字，去掉标点，不移除停用词、不做词干化。
作业中的 NLTK 分词是可选建议，本实验使用不需要下载模型的确定性正则分词。
词表仅从训练集构建，共 61329 个词项；验证集和测试集中的未知词忽略。
两种表示均自行计数并构建 CSR 稀疏矩阵，Binary 取 0/1，Frequency 保留原始计数。
两种方法共用词表、划分和分类器参数：L2 正则化，C=1.0，solver=lbfgs，max_iter=2000，tol=1e-4，fit_intercept=True，class_weight=None。
三分类使用多项逻辑回归；固定 random_state=42，并限制数值计算为单线程。
未进行超参数搜索。验证集仅用于辅助报告，不参与词表构建或训练；测试集仅用于最终评价。
收敛警告作为错误处理，不记录未收敛训练结果。

## 评价结果

Accuracy 为预测正确的样本比例；Macro-F1 为三个类别各自 F1 的算术平均，指标越大越好。

| 表示方法 | 验证 Accuracy | 验证 Macro-F1 | 测试 Accuracy | 测试 Macro-F1 | 迭代次数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binary Bag of Words | 0.985243 | 0.964315 | 0.986979 | 0.971379 | 19 |
| Word Frequency | 0.985243 | 0.965699 | 0.988715 | 0.975509 | 132 |

### Binary Bag of Words 测试集分类明细

| 类别 | Precision | Recall | F1 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| business | 0.972028 | 0.939189 | 0.955326 | 148 |
| politics | 0.947059 | 0.975758 | 0.961194 | 165 |
| sports | 0.997616 | 0.997616 | 0.997616 | 839 |

混淆矩阵（行是真实类别，列是预测类别；顺序为 business、politics、sports）：

```text
[139, 8, 1]
[3, 161, 1]
[1, 1, 837]
```

### Word Frequency 测试集分类明细

| 类别 | Precision | Recall | F1 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| business | 0.985816 | 0.939189 | 0.961938 | 148 |
| politics | 0.958333 | 0.975758 | 0.966967 | 165 |
| sports | 0.995255 | 1.000000 | 0.997622 | 839 |

混淆矩阵（行是真实类别，列是预测类别；顺序为 business、politics、sports）：

```text
[139, 7, 2]
[2, 161, 2]
[0, 0, 839]
```

## 结果分析

Binary 相对 Frequency 的测试 Accuracy 差值为 -0.1736 个百分点，Macro-F1 差值为 -0.4130 个百分点。
二值表示削弱同一新闻中重复用词的影响；原始词频保留重复次数，也会受到文档长度的影响。上述差异来自本次固定划分，不能据此断言某种表示在所有数据上都更好。
数据中的 sports 类占比较高，Accuracy 容易受到多数类表现影响，因此应同时观察 Macro-F1 和各类别明细。

## 复现与输出

在仓库根目录完成依赖安装后执行：

```bash
.venv/bin/python "NLP Hw1/task1.py"
.venv/bin/python -m unittest discover -s "NLP Hw1/tests" -v
```

- `../results/task1_metrics.json`：完整指标、环境、超参数、收敛迭代次数。
- `../results/task1_predictions.csv`：测试样本原始行索引、真实标签、两种方法的预测。
- `../results/nyt_split.json`：固定划分与原始数据指纹。
- `../results/task1_vocabulary.json`：训练集词项到列索引的映射。

本报告由实验脚本根据实际运行结果生成。当前阶段按要求提交 Markdown 记录；完整作业最终要求的 PDF 报告留待后续任务完成后统一整理。

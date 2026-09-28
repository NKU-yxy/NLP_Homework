# 自然语言处理作业

本仓库保存自然语言处理课程作业。作业一的原始要求和数据位于 `NLP Hw1/`。

## 环境配置

使用 Python 3.12（本次实际版本为 3.12.14），在仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

依赖版本固定在 `requirements.txt`。实验仅使用 CPU，无需 CUDA，也无需下载额外分词模型。

## 数据

- `NLP Hw1/data/nyt.csv`：新闻分类数据，包含 `text` 和 `label` 两列。
- `NLP Hw1/data/ag.csv`：供后续词向量任务使用，Task 1 不使用。
- `NLP Hw1/作业一要求.docx`：课程提供的作业说明。

原始数据随仓库保留；虚拟环境和缓存不提交。

## Task 1 运行方式

在仓库根目录执行：

```bash
.venv/bin/python "NLP Hw1/task1.py"
.venv/bin/python -m unittest discover -s "NLP Hw1/tests" -v
```

程序自行实现词表构建与稀疏词袋计数，分别使用二值词袋和原始词频训练多项逻辑回归。统一设置随机种子 42，按 80% / 10% / 10% 划分 NYT 训练、验证、测试集。词表仅根据训练集构建；两种方法共用相同词表、划分与分类器参数。测试集不参与拟合或调参。

输出文件：

- [Task 1 实验报告](NLP%20Hw1/reports/Task1实验结果.md)：方法、参数、Accuracy、Macro-F1、类别明细和结果分析。
- [环境配置记录](NLP%20Hw1/reports/环境配置.md)：实际环境与本机 Git 兼容说明。
- `NLP Hw1/results/nyt_split.json`：固定划分和数据指纹，供后续任务复用。
- `NLP Hw1/results/task1_metrics.json`：完整指标和参数。
- `NLP Hw1/results/task1_predictions.csv`：两种方法的逐条测试预测。
- `NLP Hw1/results/task1_vocabulary.json`：训练词表。

重复运行会验证已有划分是否与数据指纹、随机种子及划分规则一致；若不一致则报错，避免意外覆盖测试集定义。报告和评估文件由实际运行结果自动生成。原始数据行索引从 0 开始，不计表头。

可以通过 `--data`、`--output`、`--report` 指定其他输入或输出路径。默认文件位置相对于脚本目录，运行时无需切换到作业目录。当前仅完成 Task 1；完整作业的 PDF 报告待后续任务完成后统一整理。

## Git

远程仓库：<https://github.com/NKU-yxy/NLP_Homework>。

提交信息统一使用简明中文。先提交环境配置，再提交 Task 1 实现与实验结果。

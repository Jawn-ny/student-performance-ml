# Student Performance ML 实验报告

## 1. 项目目标

本项目使用 UCI Student Performance Dataset，完成一次基础但完整的机器学习实验流程。

项目包含两个任务：

### Regression

预测学生最终成绩：

```text
Target = G3
```

### Classification

根据最终成绩构造教学用分类 Target：

```text
G3 >= 10 → passed = 1
G3 < 10  → passed = 0
```

`passed` 并不是 UCI 原始数据自带的标签，而是本项目自行定义的分类目标。

---

## 2. 数据

当前使用：

```text
data/raw/student-por.csv
```

CSV 分隔符：

```text
;
```

数据规模：

```text
649 samples
33 columns
```

数据检查中未发现缺失值。

---

# 3. Regression 实验

## Experiment R1：DummyRegressor Baseline

Features：

```text
age
studytime
failures
absences
```

Target：

```text
G3
```

Train/Test Split：

```text
test_size = 0.2
random_state = 42
```

Model：

```text
DummyRegressor(strategy="mean")
```

Metric：

```text
RMSE
```

Result：

```text
RMSE ≈ 3.1726
```

DummyRegressor 基本始终预测训练集 G3 的平均值。

它的作用是建立最简单的 Regression Baseline。

---

## Experiment R2：LinearRegression

Features：

```text
age
studytime
failures
absences
```

Model：

```text
LinearRegression
```

Result：

```text
RMSE ≈ 2.8793
```

比较：

```text
DummyRegressor RMSE ≈ 3.1726
LinearRegression RMSE ≈ 2.8793
```

在当前相同实验条件和测试集下，LinearRegression 的 RMSE 更低，因此优于当前 Baseline。

但这并不说明模型已经能够准确预测真实学校环境中的学生成绩。

---

## Experiment R3：加入 G1 / G2

在原有 Features 基础上加入：

```text
G1
G2
```

Result：

```text
Without G1/G2:
RMSE ≈ 2.8793

With G1/G2:
RMSE ≈ 1.1541
```

加入 G1/G2 后，当前实验中的 Regression 误差明显下降。

但是 G1 和 G2 是较晚阶段才能获得的成绩信息。

因此：

```text
不使用 G1/G2
```

更接近较早阶段预测。

而：

```text
使用 G1/G2
```

更接近已经获得阶段性成绩后的预测。

两个实验对应的现实预测场景并不完全相同，不能简单认为 RMSE 降低完全代表模型本身变得更强。

---

# 4. Classification 实验

## Classification Target

本项目定义：

```text
G3 >= 10 → passed = 1
G3 < 10  → passed = 0
```

当前 Features：

数值特征：

```text
age
studytime
failures
absences
```

分类特征：

```text
school
sex
address
```

分类任务使用：

```text
test_size = 0.2
random_state = 42
stratify = y
```

`stratify` 用于使训练集和测试集尽量保持与原始数据相似的类别比例。

---

## Experiment C1：DummyClassifier Baseline

Model：

```text
DummyClassifier(strategy="most_frequent")
```

Result：

```text
Accuracy ≈ 0.8462
```

即约：

```text
84.62%
```

当前数据中 `passed = 1` 是多数类，因此 DummyClassifier 基本将所有学生预测为通过。

这说明较高的 Accuracy 不一定代表模型真正学到了有效规律。

---

## Experiment C2：LogisticRegression

分类特征中包含字符串，因此使用：

```text
ColumnTransformer
↓
OneHotEncoder
↓
LogisticRegression
```

并使用 Pipeline 将预处理和模型串联。

Result：

```text
Accuracy = 0.80
```

Baseline：

```text
DummyClassifier Accuracy ≈ 0.8462
```

因此，在当前 Accuracy 指标下：

```text
LogisticRegression 没有超过 Baseline
```

---

## 5. Confusion Matrix

LogisticRegression 当前结果：

```text
[[  2, 18],
 [  8,102]]
```

对应：

```text
TN = 2
FP = 18
FN = 8
TP = 102
```

真实未通过的学生共有：

```text
20
```

其中：

```text
正确识别未通过：2
错误预测为通过：18
```

真实通过的学生共有：

```text
110
```

其中：

```text
正确预测通过：102
错误预测为未通过：8
```

因此当前模型明显更倾向于预测：

```text
passed = 1
```

只看：

```text
Accuracy = 80%
```

无法发现模型对未通过学生识别能力很差的问题。

因此 Classification 不能只观察 Accuracy，还需要结合 Confusion Matrix 分析错误类型。

---

# 6. Classification G1 / G2 对照实验

Experiment A：

```text
Without G1/G2
Accuracy = 0.80

Confusion Matrix:
[[  2, 18],
 [  8,102]]
```

Experiment B：

```text
With G1/G2
Accuracy = 0.90

Confusion Matrix:
[[ 14,  6],
 [  7,103]]
```

加入 G1/G2 后：

```text
Accuracy:
0.80 → 0.90
```

对未通过学生的识别也明显改善：

```text
TN:
2 → 14

FP:
18 → 6
```

说明在当前数据和实验条件下，加入 G1/G2 后模型预测表现明显提高。

但是 G1/G2 是较晚阶段才能获得的信息，因此 Experiment B 对应的是一个更晚的预测场景。

不能简单得出：

```text
加入 G1/G2 后模型在所有场景中都更好
```

---

# 7. Preprocessing 与 Pipeline

项目使用：

```text
OneHotEncoder
ColumnTransformer
Pipeline
```

其中：

```text
OneHotEncoder
```

负责将分类 Feature 转换为模型可以处理的数值表示。

```text
ColumnTransformer
```

负责让不同类型的列使用不同的预处理方式。

```text
Pipeline
```

负责将：

```text
预处理
↓
模型
```

串成统一流程。

这样可以保证训练阶段和预测阶段采用一致的预处理规则，并降低因为手动处理流程不一致造成错误和数据泄漏的风险。

---

# 8. 模型保存

训练完成的 Classification Pipeline 保存为：

```text
models/classification_pipeline.joblib
```

保存完整 Pipeline 而不是只保存 LogisticRegression，可以同时保存：

```text
ColumnTransformer
OneHotEncoder
LogisticRegression
```

因此重新加载以后，可以直接接收原始 Features 进行预测。

已经验证：

```text
joblib.dump()
↓
joblib.load()
↓
predict()
```

能够正常工作。

需要注意：

不要加载来源不可信的 joblib / pickle 模型文件，因为反序列化可能存在安全风险。

模型文件也可能依赖：

```text
Python 版本
scikit-learn 版本
其他依赖版本
```

因此模型文件并不是永远兼容的独立文件。

---

# 9. 项目工程化

探索和学习代码主要保留在：

```text
notebooks/01_eda.ipynb
```

可以重复运行的正式代码整理到：

```text
src/
```

当前主要脚本：

```text
load_data.py
train_regression.py
train_classification.py
evaluate.py
```

职责分别为：

```text
load_data.py
→ 数据读取

train_regression.py
→ Regression 训练流程

train_classification.py
→ Classification 训练流程

evaluate.py
→ 统一评价指标
```

---

# 10. 当前项目限制

当前实验仍然存在以下限制：

- 数据集规模有限
- 第一版只使用少量 Features
- 只进行了固定的一次 Train/Test Split
- 尚未使用 Cross Validation
- 尚未系统研究过拟合和欠拟合
- 模型结果只适用于当前实验条件
- 统计关系不能直接解释为因果关系
- 当前项目仅用于机器学习教学
- 当前结果不能直接用于真实学校中的学生评价或教育决策

---

# 11. 最终总结

本项目完成了以下基本机器学习流程：

```text
CSV
↓
pandas DataFrame
↓
数据理解与 EDA
↓
Feature / Target
↓
Train / Test Split
↓
Baseline
↓
Preprocessing
↓
Pipeline
↓
Model
↓
Predict
↓
Metric
↓
Confusion Matrix
↓
G1/G2 对照实验
↓
有限实验结论
↓
正式 Python 脚本
↓
保存 Pipeline
```

本阶段最重要的认识不是寻找“最强模型”，而是：

```text
明确问题
↓
明确预测时间点
↓
判断当时可获得的 Feature
↓
严格划分训练集和测试集
↓
避免数据泄漏
↓
建立 Baseline
↓
选择合适指标
↓
分析模型具体错误
↓
谨慎解释实验结果
```
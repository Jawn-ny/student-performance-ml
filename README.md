# Student Performance ML

一个用于学习完整机器学习实验流程的学生成绩预测项目。

本项目使用 UCI Student Performance Dataset，从数据读取和探索开始，逐步完成一个基础的机器学习回归实验。

当前主要任务：

> 根据学生相关信息预测最终成绩 G3。

---

# 项目结构

```text
student-performance-ml/
│
├── data/
│   └── raw/
│       └── student-por.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── .gitignore
└── README.md
```

其中：

- `data/raw/`：保存原始数据
- `notebooks/01_eda.ipynb`：主要学习和实验 Notebook
- `README.md`：记录项目目标、实验过程和结论

`data/raw/` 已加入 `.gitignore`，原始数据不会提交到 Git 仓库。

---

# Dataset

使用 UCI Student Performance Dataset。

当前使用的数据文件：

```text
student-por.csv
```

该 CSV 使用分号 `;` 作为分隔符，因此读取方式为：

```python
df = pd.read_csv("../data/raw/student-por.csv", sep=";")
```

当前数据基本情况：

- Samples：649
- Columns：33
- Missing values：0

---

# Day 1：认识数据

第一天没有训练模型，主要目标是先理解：

> 我手里的数据到底是什么。

## 基础概念

### Dataset

Dataset 表示整个数据集。

当前的 `student-por.csv` 就是我们使用的数据集。

### Sample

Sample 表示数据集中的一条记录。

在当前项目中：

> 一名学生的一条完整记录可以看作一个 Sample。

### Feature

Feature 是提供给模型，用于进行预测的信息。

例如：

- age
- studytime
- failures
- absences

### Target

Target 是模型最终需要预测的目标。

当前回归任务的 Target：

```text
G3
```

即学生最终成绩。

### DataFrame

使用 pandas：

```python
df = pd.read_csv("../data/raw/student-por.csv", sep=";")
```

读取 CSV 后，会得到一个 DataFrame。

其中：

- `pd.read_csv(...)`：读取 CSV 文件
- 返回一个 DataFrame
- `df`：保存对这个 DataFrame 的引用

这里不是重新创建或保存一个 CSV 文件。

---

## 初步查看数据

使用：

```python
df.head()
```

查看数据前几行。

使用：

```python
df.shape
```

得到：

```text
(649, 33)
```

表示：

- 649 个 Samples
- 33 个 Columns

使用：

```python
df.columns
```

查看所有列名。

使用：

```python
df.info()
```

查看每一列的数据类型和非空数量。

当前大致包括：

- 16 个 `int64`
- 17 个 `object`

使用：

```python
df.describe()
```

查看数值数据的统计信息。

使用：

```python
df.isna().sum()
```

检查缺失值。

当前数据：

```text
没有缺失值
```

---

## G3 初步观察

使用：

```python
df["G3"].value_counts().sort_index()
```

观察不同 G3 分数对应的学生数量。

当前 G3：

- 最低值：0
- 最高值：19
- 平均值约：11.91

---

# Day 2：EDA

EDA 全称：

```text
Exploratory Data Analysis
```

即：

> 探索性数据分析。

EDA 的主要目的不是画漂亮的图，而是在训练模型前了解数据。

使用：

```python
import matplotlib.pyplot as plt
```

进行基础数据可视化。

---

## Histogram

使用 Histogram 观察 G3 的分布。

```python
fig, ax = plt.subplots()

ax.hist(df["G3"], bins=range(0, 21))

ax.set_xlabel("G3")
ax.set_ylabel("Number of students")
ax.set_title("Distribution of G3")

plt.show()
```

从当前数据中可以观察到：

> G3 主要集中在中间分数区域，10～12 附近的学生较多。

Histogram 主要适合：

> 观察数值数据的整体分布。

---

## Scatter Plot

Scatter Plot 用于观察两个数值变量之间可能存在的关系。

### absences 与 G3

观察：

```text
absences vs G3
```

可以看到：

> 较低 absences 区域的数据点较密集，而高 absences 区域的数据较少。

需要注意：

> 图上的点密集并不能直接表示比例。

多个学生可能具有完全相同的坐标，因此图中的点可能发生重叠。

---

### failures 与 G3

`failures` 是离散数值：

```text
0
1
2
3
```

因此 Scatter Plot 中会形成几条明显的垂直带。

当前可以观察到：

> failures 较高时，G3 整体上存在偏低的趋势。

但是：

> Correlation ≠ Causation

即：

> 相关关系不等于因果关系。

不能直接认为 failures 一定导致 G3 降低。

---

### studytime 与 G3

studytime 同样属于离散等级。

当前图中没有观察到特别明显的单调增长关系。

可以看到：

> studytime 较高区域中低分样本相对较少。

但同样不能直接得出因果关系。

---

## Bar Chart

先统计：

```python
g3_counts = df["G3"].value_counts().sort_index()
```

然后通过 Bar Chart 展示不同 G3 对应的学生数量。

Bar Chart 更适合：

> 比较不同离散值或者不同类别对应的数量。

Histogram 和 Bar Chart 并不是谁一定更加详细，而是适合解决的问题不同。

---

## Day 2 总结

学习了：

- Histogram：观察数值分布
- Scatter Plot：观察两个数值变量之间的关系
- Bar Chart：比较离散值或类别数量
- 图中出现关系不能直接解释为因果关系

---

# Day 3：Feature 与 Target

开始正式定义机器学习任务。

当前目标：

> 根据学生相关信息预测最终成绩 G3。

因此：

```python
y = df["G3"]
```

Target：

```text
G3
```

---

## 第一版 Features

为了先建立最简单的机器学习流程，目前只选择数值 Feature：

```python
X = df[
    [
        "age",
        "studytime",
        "failures",
        "absences"
    ]
]
```

当前 Features：

- age
- studytime
- failures
- absences

---

## 为什么第一版只选择数值 Feature

数据集中还有很多 `object` 类型的分类 Feature。

例如：

- school
- sex
- address

这些文字类型的数据暂时不能直接交给 LinearRegression。

后续会学习：

```text
OneHotEncoder
```

等方法处理分类 Feature。

因此第一版先只使用数值 Feature，把完整机器学习流程跑通。

---

## 为什么 G3 不能进入 X

因为：

```text
y = G3
```

如果 G3 同时进入 X：

> 相当于拿最终答案预测最终答案。

这种结果没有实际意义，并且属于数据泄漏问题。

---

## 为什么暂时不用 G1 和 G2

G1 和 G2 是学生前两个阶段的成绩。

当前第一版实验暂时假设：

> 希望在更早的时间点预测最终成绩。

因此暂时不使用 G1 和 G2。

这并不代表：

> G1 和 G2 永远不能使用。

后续会专门比较：

```text
不使用 G1/G2
vs
使用 G1/G2
```

分析预测时间点和 Feature 可用性对模型结果的影响。

---

## 当前 X 与 y

```text
X.shape = (649, 4)

y.shape = (649,)
```

即：

- 649 个 Samples
- 4 个 Features
- 每个 Sample 对应一个 G3 Target

---

# Day 4：Train / Test Split

机器学习模型不能只在训练数据上评价。

如果：

> 一边使用数据和答案训练，又使用同一批数据评价

就无法知道模型面对没有见过的数据时表现如何。

因此需要划分：

```text
Training Set
Testing Set
```

---

## 数据划分

导入：

```python
from sklearn.model_selection import train_test_split
```

然后：

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

当前结果：

```text
X_train: (519, 4)
X_test:  (130, 4)

y_train: (519,)
y_test:  (130,)
```

即：

- Training Samples：519
- Testing Samples：130

总计：

```text
519 + 130 = 649
```

---

## X_train

模型训练时使用的 Features。

---

## y_train

X_train 对应的真实 G3。

训练阶段：

```text
X_train + y_train
↓
fit()
```

---

## X_test

训练结束以后用于测试模型的 Features。

模型通过：

```text
X_test
↓
predict()
```

得到预测结果。

---

## y_test

X_test 对应的真实 G3。

模型在 `predict()` 时不会看到 y_test。

y_test 最后用于和模型预测结果进行比较。

可以理解成：

```text
X_test = 考试题

y_pred = 模型写出的答案

y_test = 标准答案
```

---

## test_size

```text
test_size = 0.2
```

表示：

> 大约 20% 的数据作为测试集，80% 的数据作为训练集。

---

## random_state

```text
random_state = 42
```

用于固定随机划分结果，使实验可以重复。

如果 random_state 不变：

> 每次得到的训练集和测试集基本保持一致。

如果修改：

```text
42 → 100
```

训练和测试数量基本不会变化，但具体进入训练集和测试集的学生会发生变化。

---

# Day 5：Baseline Regression

在使用真正的预测模型之前，先建立一个最简单的参考标准：

```text
Baseline
```

Baseline 的目的：

> 判断后续模型是不是真的比一个非常简单的方法更好。

---

## DummyRegressor

使用：

```python
from sklearn.dummy import DummyRegressor
```

模型：

```python
dummy_model = DummyRegressor(strategy="mean")
```

`mean` 策略：

> 学习训练集 y_train 的平均值，并对测试样本基本都预测这个平均值。

当前：

```text
y_train mean ≈ 11.79
```

因此 DummyRegressor 的预测基本都是：

```text
11.79
```

---

## fit()

使用：

```python
dummy_model.fit(X_train, y_train)
```

`fit()` 表示：

> 让当前模型从训练数据中学习。

对于 DummyRegressor：

> 主要学习 y_train 的平均值。

---

## predict()

使用：

```python
y_pred_dummy = dummy_model.predict(X_test)
```

表示：

> 使用训练完成的模型，对 X_test 进行预测。

得到：

```text
y_pred_dummy
```

---

## RMSE

使用：

```text
Root Mean Squared Error
```

评价回归模型。

RMSE 用于：

> 衡量预测值和真实值之间整体的误差大小。

RMSE：

```text
越低越好
```

理想情况：

```text
RMSE = 0
```

表示预测值与真实值完全一致。

当前 DummyRegressor：

```text
RMSE ≈ 3.17
```

这里不能理解成：

> 每一个学生都刚好预测错 3.17 分。

而是表示当前测试集整体预测误差的一个指标。

---

## Experiment 001

Problem:

```text
Regression
```

Target:

```text
G3
```

Features:

- age
- studytime
- failures
- absences

Excluded:

- G1
- G2
- G3 不作为 Feature

Train / Test:

```text
test_size = 0.2
random_state = 42

519 training samples
130 testing samples
```

Preprocessing:

```text
None
```

Model:

```text
DummyRegressor(strategy="mean")
```

Metric:

```text
RMSE
```

Result:

```text
RMSE ≈ 3.17
```

Observation:

DummyRegressor 对所有测试样本基本预测相同的训练集 G3 平均值。

Conclusion:

建立了第一个 Baseline，为后续模型提供比较标准。

Limitations:

DummyRegressor 没有根据学生不同的 Feature 做出不同预测。

---

# Day 6：LinearRegression

第一次训练真正利用 Feature 进行预测的回归模型：

```text
LinearRegression
```

---

## DummyRegressor 与 LinearRegression

DummyRegressor：

> 主要学习训练集 Target 的平均值。

LinearRegression：

> 会根据不同 Feature 学习不同的权重，并利用这些信息进行预测。

---

## 创建模型

```python
linear_model = LinearRegression()
```

这里只是：

> 创建 LinearRegression 模型对象。

此时还没有学习数据。

因此：

```text
创建模型 ≠ 训练模型
```

---

## 模型训练

使用：

```python
linear_model.fit(X_train, y_train)
```

模型会从训练数据中学习：

```text
coef_
intercept_
```

---

## coef_

当前 Feature 顺序：

```text
age
studytime
failures
absences
```

模型学习到的系数约为：

```text
age        → +0.0067
studytime  → +0.7648
failures   → -2.0302
absences   → -0.0190
```

这些系数：

> 是模型通过 fit() 从训练数据中学习得到的。

不是人为提前指定的。

例如：

```text
studytime coefficient > 0
```

说明：

> 在当前线性模型中，studytime 与预测 G3 存在正向线性关系。

但是：

> 模型中的统计关系不等于因果关系。

不能直接说：

> studytime 增加一定会导致 G3 增加。

---

## intercept_

当前：

```text
intercept ≈ 10.7554
```

目前可以简单理解成：

> 线性预测公式中的基础值。

暂时不深入学习数学推导。

---

## Predict

使用：

```python
y_pred_linear = linear_model.predict(X_test)
```

LinearRegression 会根据不同学生的 Feature 给出不同预测值。

例如：

```text
12.33
12.35
12.40
6.12
...
```

预测结果出现小数是正常的。

因为：

> Regression 模型可以预测连续数值。

---

## LinearRegression RMSE

当前：

```text
LinearRegression RMSE ≈ 2.88
```

Baseline：

```text
DummyRegressor RMSE ≈ 3.17
```

比较：

| Model | RMSE |
| --- | ---: |
| DummyRegressor | 3.17 |
| LinearRegression | 2.88 |

因为：

```text
2.88 < 3.17
```

所以：

> 在当前实验条件和相同测试集上，LinearRegression 的预测误差低于 DummyRegressor。

因此目前可以认为：

> LinearRegression 在本次实验条件下优于 Baseline。

但不能直接认为：

> LinearRegression 已经是一个非常准确的预测模型。

---

## Experiment 002

Problem:

```text
Regression
```

Target:

```text
G3
```

Features:

- age
- studytime
- failures
- absences

Excluded:

- G1
- G2
- G3 不作为 Feature

Train / Test:

```text
test_size = 0.2
random_state = 42

519 training samples
130 testing samples
```

Preprocessing:

```text
None
```

Model:

```text
LinearRegression
```

Metric:

```text
RMSE
```

Result:

```text
RMSE ≈ 2.88
```

Baseline:

```text
DummyRegressor RMSE ≈ 3.17
```

Observation:

LinearRegression 会利用不同 Feature 学习不同权重，并针对不同测试样本产生不同预测结果。

Conclusion:

> 在当前实验设置和相同测试集下，LinearRegression 的 RMSE 低于 DummyRegressor，因此当前表现优于 Baseline。

Limitations:

- 目前只使用 4 个数值 Feature
- 暂未加入分类 Feature
- 只进行了一次 Train / Test Split
- 当前统计关系不能直接解释为因果关系

---

# Day 7：第一周总结

第一周完成了第一个完整的机器学习 Regression 实验闭环。

---

## 完整实验流程

```text
CSV
↓
pandas 读取 DataFrame
↓
数据理解与 EDA
↓
明确预测问题
↓
选择 Feature X 和 Target y
↓
Train / Test Split
↓
建立 Baseline
↓
训练 LinearRegression
↓
Predict
↓
RMSE
↓
模型比较
↓
得出有限实验结论
```

---

## Train 与 Test 的关系

训练阶段：

```text
X_train + y_train
↓
fit()
↓
训练好的模型
```

预测阶段：

```text
X_test
↓
predict()
↓
y_pred
```

评价阶段：

```text
y_pred
vs
y_test
↓
RMSE
```

需要重点区分：

```text
X_test = 模型测试时看到的 Features

y_pred = 模型预测出来的结果

y_test = 数据中真实存在的结果
```

模型在：

```text
predict()
```

时不会看到 `y_test`。

---

## Result 与 Conclusion

当前实验 Result：

```text
DummyRegressor RMSE ≈ 3.17

LinearRegression RMSE ≈ 2.88
```

Result 表示：

> 客观得到的实验数字。

Conclusion 表示：

> 根据实验结果进行的解释。

RMSE 越低表示当前测试集上的整体预测误差越小。

因此：

> LinearRegression 当前测试集上的预测误差低于 DummyRegressor。

---

## Baseline 的意义

Baseline 并不是为了获得最好的模型效果。

它的作用是：

> 为后面的模型提供一个参考标准。

例如：

```text
DummyRegressor RMSE = 3.17
LinearRegression RMSE = 3.50
```

因为：

```text
3.50 > 3.17
```

说明：

> 当前 LinearRegression 没有优于最简单的 Baseline。

此时应该进一步分析：

- Feature 是否提供足够信息
- 模型是否适合当前数据
- Train / Test Split 是否产生影响
- 数据是否存在问题
- 实验设置是否合理

不能因为模型更加复杂，就默认模型一定更加有效。

---

## 当前实验结论

当前结果：

```text
DummyRegressor RMSE ≈ 3.17
LinearRegression RMSE ≈ 2.88
```

因此：

> 在使用相同 Features、相同 Train / Test Split 和相同评价指标的条件下，LinearRegression 当前测试集的预测误差低于 DummyRegressor。

目前可以认为：

> LinearRegression 在本次实验条件下优于 Baseline。

但是不能直接认为：

- LinearRegression 已经是非常准确的学生成绩预测模型
- Feature 与 G3 之间存在因果关系
- 当前模型可以直接用于真实学校环境

---

## 当前限制

目前实验仍然比较简单：

- 只使用了 4 个数值 Feature
- 暂未使用分类 Feature
- 暂未加入 G1 和 G2
- 只进行了一次 Train / Test Split
- 当前数据集规模有限
- 暂未进行更多稳定性验证
- 模型中的统计关系不能直接解释为因果关系
- 当前实验结果不能直接推广到真实学校环境

---

# 第一周完成情况

- [x] 读取 CSV 数据
- [x] 理解 Dataset / Sample / Feature / Target
- [x] pandas DataFrame 基础
- [x] 检查数据结构
- [x] 检查缺失值
- [x] EDA
- [x] Histogram
- [x] Scatter Plot
- [x] Bar Chart
- [x] Feature / Target
- [x] Train / Test Split
- [x] DummyRegressor Baseline
- [x] RMSE
- [x] LinearRegression
- [x] fit / predict
- [x] coef_ / intercept_
- [x] 模型比较
- [x] 第一周实验总结

---

# 当前进度

已经完成：

> 第一个完整的 Regression 机器学习实验闭环。

当前流程：

```text
数据
→ EDA
→ X / y
→ Train / Test Split
→ Baseline
→ LinearRegression
→ Predict
→ RMSE
→ 模型比较
→ 实验结论
```

下一阶段将开始学习：

> Categorical Features 与 OneHotEncoder

## Day 8：Categorical Features 与 One-Hot Encoding

本阶段开始处理分类特征（Categorical Features）。

数据集中包含 `school`、`sex`、`address` 等文字类别特征。这些类别不能简单编码为 `1、2、3` 后直接作为普通数值特征，因为这样可能人为引入不存在的大小和距离关系。

学习并实践了 One-Hot Encoding：

* 每个类别转换为独立的 0/1 特征
* `1` 表示样本属于该类别
* `0` 表示样本不属于该类别
* 避免人为制造类别之间的大小顺序

使用 sklearn 的 `OneHotEncoder` 对 `school` 进行实验。

确认：

* `school` 包含 `GP` 和 `MS` 两个类别
* 编码后得到 `school_GP` 和 `school_MS`
* `GP → [1, 0]`
* `MS → [0, 1]`

进一步理解：

* `fit()`：学习数据中存在的类别和编码规则
* `transform()`：按照已经学习的规则转换数据
* OneHotEncoder 默认可以使用稀疏矩阵保存编码结果
* 测试数据应该使用训练阶段学习到的预处理规则，而不是重新 `fit`

## Day 9：ColumnTransformer 与 Pipeline

本阶段学习如何同时处理数值特征和分类特征，并将预处理流程与模型串联起来。

当前使用的数值特征：

* age
* studytime
* failures
* absences

当前使用的分类特征：

* school
* sex
* address

使用 `ColumnTransformer` 对不同类型的特征应用不同处理方式：

* 数值特征使用 `passthrough`，保持原样
* 分类特征使用 `OneHotEncoder`

原始混合特征共有 7 列：

* 4 个数值特征
* 3 个分类特征

经过 One-Hot Encoding 后，分类特征被展开，因此转换后的特征数量增加到 10 列。

进一步使用 `Pipeline` 将：

* ColumnTransformer 预处理
* LinearRegression 模型

串成一个完整流程。

理解：

* `ColumnTransformer` 负责不同列使用不同预处理方式
* `Pipeline` 负责将多个步骤按顺序串联
* `LinearRegression` 负责学习特征与 G3 之间的线性关系

使用：

* `pipeline.fit()`：执行训练阶段的预处理并训练模型
* `pipeline.predict()`：使用训练阶段已学习的预处理规则处理测试集并进行预测

Pipeline 可以减少重复手动操作，并降低训练与预测流程不一致带来的人为错误。


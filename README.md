# Student Performance ML

学生成绩数据分析与机器学习实验项目。

本项目使用 Python 进行数据分析，并逐步学习完整的机器学习实验流程：

数据理解 → 探索性数据分析 → Feature / Target → Train / Test Split → Baseline → 模型训练 → 预测 → 评价指标 → 错误分析 → 实验结论。

当前阶段的目标是理解规范的机器学习实验流程，而不是追求复杂模型或最高预测准确率。

## 数据来源

数据来自 UCI Machine Learning Repository：

**Student Performance Dataset**

当前使用：

`student-por.csv`

该文件记录 Portuguese language 课程相关的学生数据。

---

## Day 1：认识数据

已完成：

* 了解 Dataset、Sample、Feature、Target 和 DataFrame
* 使用 pandas 读取 `student-por.csv`
* 查看数据前几行
* 查看 DataFrame 的行数和列数
* 查看字段名称
* 查看字段的数据类型
* 查看数值字段的基本统计信息
* 检查缺失值
* 初步查看 `G3`

当前数据规模：

* 649 条记录
* 33 个字段

数据类型：

* 16 个 `int64` 字段
* 17 个 `object` 字段

缺失值检查：

* 当前数据中未发现缺失值

当前回归任务计划预测：

`G3` —— 学生最终成绩。

---

## Day 2：探索性数据分析 EDA

学习了探索性数据分析（EDA）的基本思想：

> 在训练模型之前，通过统计结果和图表理解数据的分布、特点和变量之间可能存在的关系。

学习并使用 Matplotlib 绘制了：

### G3 Histogram

用于观察最终成绩 `G3` 的整体分布。

学习内容：

* Histogram
* bins
* Figure
* Axes
* xlabel
* ylabel
* title

观察到：

* G3 主要集中在中间成绩范围
* 某些较低成绩区间中的数据非常少

### Absences vs G3

使用 Scatter Plot 观察：

`absences` 与 `G3`

初步观察：

* 低缺勤区域的数据记录明显更多
* 高缺勤记录相对较少
* 高缺勤区域仍然同时存在较高和较低的 G3

当前不能仅根据散点图得出：

“缺勤导致成绩下降。”

### Failures vs G3

使用 Scatter Plot 观察：

`failures` 与 `G3`

初步观察：

* failures 只有少数离散取值，因此散点形成多条竖带
* failures 较高的记录中，G3 看起来整体偏低

当前只能认为存在可能的关联，不能直接认为存在因果关系。

### Studytime vs G3

使用 Scatter Plot 观察：

`studytime` 与 `G3`

初步观察：

* studytime 是离散变量，因此形成多条竖带
* 没有观察到非常明显的“studytime 越高，G3 就持续越高”的趋势
* 较高 studytime 的记录中似乎较少出现特别低的 G3

以上都只是当前数据中的观察，不能直接解释为因果关系。

### G3 Bar Chart

先通过 `value_counts()` 统计每一个 G3 成绩对应的学生数量，再使用 Bar Chart 进行比较。

进一步理解了：

* Histogram：观察数值变量整体分布
* Bar：比较不同类别或离散取值
* Scatter：观察两个数值变量之间的关系

核心原则：

**相关不代表因果。**

---

## Day 3：Feature 与 Target

进一步学习了 Feature、Target、`X` 和 `y` 的含义。

当前回归任务：

```text
Target = G3
```

创建：

```text
X = 模型可以看到的 Features
y = 模型希望预测的 Target
```

第一版选择的 Features：

* `age`
* `studytime`
* `failures`
* `absences`

当前：

```text
X.shape = (649, 4)
y.shape = (649,)
```

表示：

* 共有 649 个样本
* 每个样本当前使用 4 个 Feature
* 每个样本对应一个 G3 Target

当前 X 中的 4 个 Feature 均为数值类型。

### 为什么 G3 不能进入 X

因为 G3 本身就是当前预测目标。

如果同时把 G3 放入 X，就会变成：

“使用 G3 预测 G3”

这不具有实际预测意义。

### 为什么第一版暂时不使用 G1 和 G2

* G1：第一阶段成绩
* G2：第二阶段成绩
* G3：最终成绩

当前第一版实验假设模型用于较早阶段预测最终成绩。

在这个时间点，G1 和 G2 可能还没有产生，因此暂时不作为 Feature。

这并不意味着 G1 和 G2 永远不能使用。

如果预测场景改为：

“G2 已经公布以后预测最终 G3”

那么 G1 和 G2 就可能成为合理的 Feature。

因此：

> 一个 Feature 能不能使用，不仅取决于数据集中有没有它，还取决于实际预测发生时这个信息是否能够获得。

### 为什么第一版只使用少量数值 Feature

当前目标是先理解最基本的机器学习数据流程：

```text
DataFrame
↓
Feature / Target
↓
X / y
```

暂时不加入大量字符串类别变量，以避免同时引入 OneHotEncoder、ColumnTransformer 等新的预处理内容。

之后会逐步增加类别 Feature，并建立更规范的机器学习 Pipeline。

---

## 当前学习原则

机器学习实验不能从选择模型开始。

当前流程：

数据来源
↓
理解数据
↓
检查数据质量
↓
探索性数据分析
↓
明确预测目标
↓
明确预测发生的时间
↓
确定当时真正能够获得的 Feature
↓
构造 X / y
↓
训练集 / 测试集
↓
Baseline
↓
模型训练与评价

当前特别需要注意：

* Target 不能进入 Feature
* Feature 是否合理取决于实际预测场景
* 相关不代表因果
* 模型指标高并不自动代表实验设计合理

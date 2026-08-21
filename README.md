# Student Performance ML

学生成绩数据分析与机器学习实验项目。

本项目使用 Python 进行数据分析，并逐步学习完整的机器学习实验流程，包括：

数据理解 → 探索性数据分析 → Feature / Target → Train / Test Split → Baseline → 模型训练 → 预测 → 评价指标 → 错误分析 → 实验结论。

当前阶段以理解机器学习实验流程为主要目标，不追求复杂模型或最高预测准确率。

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

* studytime 也是离散变量，因此形成多条竖带
* 没有观察到非常明显的“studytime 越高，G3 就持续越高”的趋势
* 较高 studytime 的记录中似乎较少出现特别低的 G3

以上都只是当前数据中的观察，不能直接解释为因果关系。

### G3 Bar Chart

先通过 `value_counts()` 统计每一个 G3 成绩对应的学生数量，再使用 Bar Chart 进行比较。

进一步理解了：

* Histogram：观察数值变量整体分布
* Bar：比较不同类别或离散取值
* Scatter：观察两个数值变量之间的关系

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
确定预测时真正可获得的 Feature
↓
训练与评价模型

同时需要始终注意：

**相关不代表因果。**

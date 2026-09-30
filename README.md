# SignalForge

**量化研究与市场结构分析框架**

SignalForge 是一个持续开发中的 Python 量化研究项目，主要用于将历史市场数据转化为**结构化、可测试、可复现的研究流程**。

目前项目重点围绕：

* 市场数据处理与校验
* 量价关系研究
* 市场结构分析
* 市场状态转变检测
* Research Feature / Label 构建
* 研究验证
* 风险指标与评估
* Paper Trading

> 🚧 **项目状态：持续开发中（Active Development）**
> SignalForge 不是一个已经完成的交易系统，而是一个会持续迭代的数据与量化研究框架。

---

## 一、项目定位

SignalForge 当前的核心思路是：

> **先做好研究，再考虑交易执行。**

项目并不以“自动产生买卖信号”为第一目标，而是尝试建立一套从原始市场数据到研究结论的完整流程：

市场数据
↓
数据校验
↓
特征与标签构建
↓
量价关系分析
↓
市场结构分析
↓
市场状态转变检测
↓
研究验证
↓
风险评估
↓
策略研究

当前项目主要用于**量化研究与模拟交易实验**，实盘交易功能暂未启用。

---

## 二、当前已实现

### 1. 市场数据

目前项目已经建立基础的历史市场数据处理能力，包括：

* OHLCV 数据处理
* 数据格式与有效性校验
* 数据来源记录
* 数据 freshness 信息
* 历史行情数据处理

数据层的设计目标是让研究所使用的数据更加明确、可追踪。

---

### 2. 量化研究框架

项目已经建立基础 Research Pipeline，用于将市场数据进一步转化为研究数据。

目前包括：

* Research Condition Comparison
* Basic Feature Construction
* Future Return Labels
* Research Data Split
* Purged Research Data Split
* Research Validation Framework
* Risk Metrics
* Research Reporting

其中，**Feature、Future Return Label 与 Validation Data 的分离**，用于降低研究过程中发生未来信息泄露（Data Leakage）的风险。

---

### 3. 量价关系研究

SignalForge 当前的研究方向之一是：

> **从价格与成交量之间的关系中寻找可以被量化和验证的市场特征。**

目前重点关注：

* Price Movement
* Volume Changes
* Volume-Price Relationships
* Market Activity
* Price-Volume Structure

这里的量价分析并不是简单依赖某一个技术指标产生交易信号，而是希望将市场行为转化为可以进入研究流程的结构化变量。

---

### 4. 市场结构分析

项目目前已经加入独立的 Market Structure 模块：

research/
└── structure/
　　├── **init**.py
　　└── analyzer.py

该模块用于对市场数据中的结构特征进行分析，为后续研究提供更加明确的结构化变量。

---

### 5. 市场状态转变检测

项目同时加入 Market Transition 模块：

research/
└── transition/
　　├── **init**.py
　　└── detector.py

主要用于研究不同市场状态之间的变化以及可能存在的结构性转变。

需要说明的是：

> 当前这些模块属于**研究工具**，并不意味着已经形成经过充分验证的预测模型或盈利策略。

---

### 6. Research Validation & Risk

SignalForge 不仅关注历史收益，也加入了研究验证与风险分析框架。

目前包括：

* Validation Split
* Purged Research Split
* Risk Metrics
* Condition Comparison
* Research Reporting

项目希望逐步回答的不是：

> “这个条件历史上赚不赚钱？”

而是：

> **“这个市场关系是否稳定、可重复，并且具有进一步研究的价值？”**

---

## 三、测试

项目使用 `pytest` 进行自动化测试。

当前版本：

**265 tests passed**

运行测试：

`python -m pytest -q`

随着项目继续开发，测试数量也会持续增加。

---

## 四、项目结构

SignalForge/
├── data/
│
├── research/
│　　├── structure/
│　　│　　├── **init**.py
│　　│　　└── analyzer.py
│　　│
│　　├── transition/
│　　│　　├── **init**.py
│　　│　　└── detector.py
│　　│
│　　└── pipeline.py
│
├── tests/
│　　├── test_structure_analyzer.py
│　　└── test_transition_detector.py
│
└── README.md

项目结构会随着后续研究模块的增加持续调整。

---

## 五、设计原则

### Research First

目前优先建设研究基础设施，而不是直接连接真实资金进行交易。

### 数据与行为优先

当前研究重点放在可以直接从市场数据观察和计算的关系上，包括：

* Price
* Volume
* Market Structure
* Market Transition

技术指标可以作为研究工具，但不会被默认视为有效交易逻辑。

### 可测试

研究逻辑尽可能通过明确的输入、输出和自动化测试进行验证。

### 可复现

研究过程中的数据处理、特征构建、标签定义和验证方法应尽可能明确，使研究结果能够被重复。

### 风险意识

策略研究不仅关注收益，也关注：

* 风险
* 样本外表现
* 数据泄露
* 稳健性
* 不同市场条件下的表现

---

## 六、当前边界

SignalForge 当前定位为：

**量化研究 + Paper Trading 项目。**

目前**不包括**：

* ❌ 真实资金交易
* ❌ PTrade 接入
* ❌ Broker Execution
* ❌ 高频交易基础设施
* ❌ 已验证的自动盈利策略

这些内容是否加入，将根据后续研究结果和项目发展逐步决定。

---

## 七、后续计划

SignalForge 会持续开发和更新。

### Research

* [ ] 增加更多量价研究特征
* [ ] 扩展 Market Structure 分析
* [ ] 扩展 Market Transition Detection
* [ ] 增加更多 Research Conditions
* [ ] 完善 Research Validation
* [ ] 加强 Out-of-Sample Evaluation
* [ ] 扩展 Risk Analysis

### Data

* [ ] 更完善的历史行情数据管线
* [ ] 增加更多数据源
* [ ] 完善数据质量检查
* [ ] 扩展不同市场的数据支持

### Strategy

* [ ] 基于研究结果构建候选策略
* [ ] 策略之间的系统化比较
* [ ] Paper Trading 持续完善
* [ ] 更严格的策略评估

### Future

后续是否进入实盘交易、PTrade 等执行层，将在研究框架进一步成熟后再决定。

---

## 八、开发状态

SignalForge 是一个**持续更新中的个人量化研究项目**。

目前的版本并不是项目终点。

随着新的研究假设、数据、测试结果和方法加入，项目中的模块、研究结论和实现方式都会持续调整。

项目会优先保证：

**数据可靠 → 逻辑明确 → 测试充分 → 验证严格 → 再进行策略扩展。**

---

## License

项目目前处于持续开发阶段，License 将在项目进一步稳定后确定。

---

**SignalForge**

*量化研究 · 市场结构 · 持续迭代*

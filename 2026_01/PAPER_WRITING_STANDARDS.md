# 论文编写与发布规范 (2026版)

> **创建日期**: 2026-01-10
> **基于**: 50+ Multi-CLI研究轮次, 200+ 信息源检索
> **适用范围**: AI/ML学术论文, 技术报告

---

## 1. 作者信息标准化

### 1.1 推荐姓名格式

| 场景 | 格式 | 示例 |
|------|------|------|
| **国际期刊/会议** | Western Order | **Jiqiang Feng** |
| **arXiv投稿** | Western Order | **Jiqiang Feng** |
| **中文期刊** | 中文名 | **冯继强** |
| **引用格式** | Last, First | Feng, Jiqiang |

**选择理由**（基于研究）:
1. Google Scholar、Semantic Scholar等数据库默认按Western Order索引
2. 著名案例：李飞飞 → "Fei-Fei Li"（国际发表）
3. ORCID可统一关联不同命名形式
4. 一致性最重要，选定后保持不变

### 1.2 邮箱选择优先级

| 邮箱类型 | 地址 | arXiv背书 | 永久性 | 推荐度 |
|----------|------|-----------|--------|--------|
| 教育机构 | `jf2563@nau.edu` | ✅ 加速 | ⚠️ 毕业失效 | ⭐⭐⭐⭐⭐ |
| 公司域名 | `feng@innora.ai` | ✅ 可接受 | ✅ 自控 | ⭐⭐⭐⭐ |
| 个人Gmail | `met3or@gmail.com` | ⚠️ 较慢 | ✅ 永久 | ⭐⭐⭐ |

**首次arXiv投稿推荐**: `jf2563@nau.edu`
- .edu邮箱加速endorsement获取
- 配合ORCID确保长期可追溯性

### 1.3 ORCID设置（强烈推荐）

```
ORCID: https://orcid.org/[your-id]
作用: 统一学术身份，跨平台关联所有发表
设置: 添加所有邮箱变体和姓名形式
```

---

## 2. arXiv投稿规范 (2026最新)

### 2.1 分类选择

| 论文类型 | 主分类 | 交叉分类 |
|----------|--------|----------|
| AI Agent自迭代 | `cs.AI` | `cs.SE`, `cs.MA` |
| LLM优化 | `cs.CL` | `cs.LG`, `cs.AI` |
| 强化学习 | `cs.LG` | `cs.AI` |
| 安全攻防 | `cs.CR` | `cs.AI`, `cs.LG` |

### 2.2 格式要求

**文件格式优先级**:
1. **LaTeX** (推荐) - 最佳渲染效果
2. **PDF** (可接受) - 需满足特定要求

**LaTeX图片格式**:
- PDFLaTeX: `.pdf`, `.jpg`, `.png`
- 禁止: `.eps` (除非用LaTeX编译)

**摘要长度**: 150-250词（英文）

### 2.3 2026年新政策

⚠️ **重要变更**:
1. **综述类论文** (cs.*) 需先通过同行评审才能提交
2. **2026年2月起**: 所有投稿需完整英文版本
3. **首次投稿**: 需获得endorsement（.edu邮箱加速）

### 2.4 许可证选择

| 许可证 | 商用 | 修改 | 推荐场景 |
|--------|------|------|----------|
| CC BY 4.0 | ✅ | ✅ | 最大传播 |
| CC BY-SA 4.0 | ✅ | ✅ | 保持开放 |
| CC BY-NC-ND | ❌ | ❌ | 保护性强 |
| arXiv perpetual | - | - | 默认选项 |

---

## 3. 论文结构规范

### 3.1 标准结构

```
1. Title
2. Authors and Affiliations
3. Abstract (150-250 words)
4. Introduction
5. Related Work
6. Methodology / Technical Approach
7. Experiments and Evaluation
8. Results and Discussion
9. Limitations
10. Conclusion
11. Acknowledgments (optional)
12. References
13. Appendix (optional)
```

### 3.2 各部分要求

#### Abstract (摘要)
- **长度**: 150-250词
- **结构**: 问题 → 方法 → 结果 → 影响
- **禁止**: 引用、缩写首次出现、方程

#### Introduction (引言)
- **段落1**: 研究背景和动机
- **段落2**: 现有方法的局限性
- **段落3**: 本文贡献（使用bullet points）
- **段落4**: 论文组织结构

#### Related Work (相关工作)
- 按主题分组，非按时间
- 明确说明与本文的区别
- 引用最新文献（2024-2026）

#### Methodology (方法)
- 清晰的架构图
- 算法伪代码（Algorithm环境）
- 数学符号一致性

#### Experiments (实验)
- **数据集**: 公开可获取
- **基线**: 包含SOTA方法
- **指标**: 标准评估指标
- **消融实验**: 验证各组件贡献

#### Limitations (局限性)
- 诚实描述方法局限
- 讨论适用场景边界
- 提出未来改进方向

---

## 4. 引用规范

### 4.1 必引论文清单 (AI Agent自迭代主题)

**核心论文 (2025-2026)**:
1. [2508.07407] A Comprehensive Survey of Self-Evolving AI Agents (Aug 2025)
2. [2510.07841] Self-Improving LLM Agents at Test-Time (Oct 2025)
3. [2512.17102] RL for Self-Improving Agent with Skill Library (Dec 2025)
4. [2504.15228] A Self-Improving Coding Agent (Apr 2025)
5. [2512.02731] Self-Improving AI Agents through Self-Play (Dec 2025)
6. [2506.05109] Intrinsic Metacognitive Learning for Self-Improving Agents (Jun 2025)

**经典基础论文**:
- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)
- Chain-of-Thought Prompting (Wei et al., 2022)
- Self-Instruct: Aligning LMs with Self-Generated Instructions (2023)

### 4.2 引用格式

**文中引用**:
- 数字格式: `[1]`, `[1,2]`, `[1-3]`
- 作者年份: `Feng et al. (2026)`, `(Feng et al., 2026)`

**参考文献格式**:
```bibtex
@article{feng2026selfiteration,
  title={AI Agent Self-Iteration System: Architecture and Implementation},
  author={Feng, Jiqiang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 5. 写作质量检查清单

### 5.1 提交前检查

- [ ] 拼写检查完成（Grammarly/LanguageTool）
- [ ] 图表编号连续且被引用
- [ ] 所有缩写首次出现时解释
- [ ] 数学符号前后一致
- [ ] 参考文献完整且格式统一
- [ ] 代码/数据可用性声明
- [ ] 伦理声明（如适用）
- [ ] 摘要在150-250词范围内
- [ ] 作者信息正确

### 5.2 常见错误避免

❌ **避免**:
- 使用"I"/"We"过度（学术论文使用被动语态）
- 模糊表达："some", "many", "various"
- 过度claim："novel", "first", "best"
- 引用不足或过度引用

✅ **推荐**:
- 具体数据支撑观点
- 明确定义术语
- 逻辑清晰的论证链
- 适度使用量化表述

---

## 6. 投稿流程检查表

### 6.1 arXiv投稿步骤

```
□ Step 1: 注册arXiv账号（需.edu邮箱加速）
□ Step 2: 获取endorsement（首次投稿）
□ Step 3: 准备源文件（LaTeX或PDF）
□ Step 4: 选择分类（cs.AI + 交叉分类）
□ Step 5: 填写元数据（Title, Abstract, Authors）
□ Step 6: 选择许可证（推荐CC BY 4.0）
□ Step 7: 上传并预览
□ Step 8: 提交并等待处理（通常24-48小时）
□ Step 9: 获取arXiv ID后更新ORCID
```

### 6.2 版本管理

- v1: 初始版本
- v2+: 修订版本（需说明主要变更）
- 建议：重大更新才提交新版本

---

## 7. 工具推荐

### 7.1 写作工具
- **Overleaf**: 在线LaTeX协作
- **Grammarly**: 语法检查
- **Writefull**: 学术写作辅助
- **Connected Papers**: 文献发现

### 7.2 格式检查
- **arXiv Compiler**: 本地预编译测试
- **latexdiff**: 版本差异对比
- **BibTeX Tidy**: 参考文献格式化

---

**创建时间**: 2026-01-10 13:30:00 +0800
**版本**: v1.0
**适用项目**: pubilie_doc论文发布流程

# CLAUDE.md - Innora Insights 项目配置

> **项目**: Innora Insights Newsletter
> **Substack**: andy0feng.substack.com
> **创建时间**: 2026-01-12
> **版本**: v2.0（双语+双主题版）

---

## 📌 项目核心信息

### 项目定位
- **品牌名**: Innora Insights
- **Tagline**: "Where AI Security Meets Intelligent Automation"
- **主题**: AI安全 + 智能自动化 + 数字孪生 + 超级个体方法论
- **目标读者**:
  - 🔒 AI安全从业者、安全工程师
  - 🤖 AI构建者、自动化爱好者
  - 🚀 技术决策者、创业者
- **发布频率**: 每周一期（Tuesday发布）
- **平台**: Substack (主) + 微信公众号 (辅)

### 内容风格
- **语言**: 双语并行（English + 中文，每个章节两种语言）
- **风格**: 专业深度、数据驱动、实战导向
- **结构**: 双语开篇 → 5个要点双语 → 工具推荐 → 数据角 → 预告

### 定价策略
| 层级 | 价格 | 内容 |
|------|------|------|
| Free | $0 | 周刊Newsletter |
| Paid | $8/月 或 $80/年 | 深度分析 + 工具折扣 |
| Founding | $240/年 | 以上全部 + 1v1咨询30分钟 |

---

## 🎯 双语写作指南

### Newsletter结构模板（Bilingual）

```markdown
# 📮 Innora Insights Issue #XX

> **[English Title]**
> **[中文标题]**

---

## 👋 Welcome / 欢迎
[EN] English introduction (3-5 sentences)
[中文] 中文介绍（3-5句）

## 📌 This Week's Key Insights / 本周要点

### 1. [Topic Title / 话题标题]
[EN] English content (150-300 words)
[中文] 中文内容（150-300字）
🔑 Key Takeaway: [一句话总结]

[重复3-5个要点...]

## 🛠 Tool of the Week / 本周工具
[工具介绍，含代码示例]

## 📊 Data Corner / 数据角
[数据可视化，ASCII图表]

## 👀 Coming Next Week / 下期预告
[EN] Bullet points
[中文] 要点列表

## 📬 Let's Connect / 保持联系
[双语CTA]
```

### 写作规范
1. **双语平衡**: 每个章节English和中文内容量对等
2. **Hook强度**: 开篇必须在3句内抓住读者（两种语言都要）
3. **数据支撑**: 每个论点至少1个数据来源
4. **代码示例**: 每期至少1个可执行的代码/命令示例
5. **长度控制**: 4000-6000字（含双语，阅读时间10-15分钟）

### 内容主题矩阵

| 主题领域 | 频率 | 示例话题 |
|----------|------|----------|
| **AI Security** | 每期必含 | 提示注入、Agent安全、LLM攻击面 |
| **Automation** | 每期必含 | n8n工作流、Claude Code、自动化脚本 |
| **Digital Twins** | 双周一次 | 安全测试、设备模拟、虚拟环境 |
| **Super-Individual** | 月度一次 | AI赋能、效率提升、个人品牌 |

---

## 📁 关键文件路径

| 用途 | 路径 |
|------|------|
| Newsletter期刊 | `content/issues/` |
| Lead Magnets | `lead-magnets/` |
| 内容模板 | `templates/` |
| 自动化工作流 | `automation/` |
| 项目文档 | `docs/` |
| Substack资源 | `assets/substack/` |

---

## 🔗 关联资源

| 资源 | 链接/路径 | 用途 |
|------|-----------|------|
| **Substack** | andy0feng.substack.com | 主发布平台 |
| **pubilie_doc** | `~/Documents/code/pubilie_doc/` | AI安全文章来源 |
| **5year** | `~/Documents/code/2026/5year/` | 总体规划 |
| **n8n** | 192.168.80.2:5678 | 自动化工作流 |
| **OmniSec** | `~/Documents/code/company/OmniSec/` | 安全工具代码 |

---

## ✅ 发布检查清单

发布前必须确认：
- [ ] 双语内容完整（每个章节都有EN+中文）
- [ ] 标题吸引力（英文+中文都能3秒抓住注意力）
- [ ] 要点清晰（5个要点是否一目了然）
- [ ] 数据准确（来源是否可靠，已注明Source）
- [ ] 代码可运行（示例代码已测试）
- [ ] 链接有效（所有链接是否可点击）
- [ ] CTA明确（读者下一步行动是什么）
- [ ] Substack格式（粘贴后格式正确）

---

## 📧 Substack设置

### Welcome Email配置
- 触发: 新订阅者
- 内容: 自我介绍 + Lead Magnet链接 + 内容预期

### 付费升级触发
- 阅读5期后显示付费内容预览
- 年付享2个月免费

---

**最后更新**: 2026-01-12
**版本**: v2.0

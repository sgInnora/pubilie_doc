# Multi-CLI协作架构：构建多AI协同工作流

> **作者**: Innora Security Research Team
> **发布日期**: 2026年1月10日
> **阅读时长**: 约12分钟
> **技术难度**: 中高级

---

## 执行摘要

单一AI模型无法覆盖所有场景。Claude擅长架构设计和深度分析，Gemini具备强大的联网搜索能力，Codex在代码审查和重构方面表现出色。如何让这三个工具协同工作？

本文介绍我们设计的Multi-CLI协作框架。通过HANDOFF协议实现上下文的结构化传递，5阶段工作流保证任务有序执行。实测表明，这套系统在项目分析任务中将效率提升了3倍以上。

你将获得：可直接部署的Bash脚本、经过验证的协议设计、真实项目的实践经验。

---

## 问题场景

我最初只用Claude Code处理开发任务。效果不错，但很快遇到了瓶颈。

有次需要调研一个新框架的最新版本特性。Claude的知识截止在2025年5月，无法提供2026年的更新内容。我只能手动搜索、整理、再喂给Claude分析。这个过程耗时且容易遗漏关键信息。

另一次，我需要对一个10万行的代码库做全面审查。Claude处理到一半就开始遗忘前面的上下文。我不得不分多次提交，然后手动合并结果。审查质量参差不齐。

还有些任务需要执行实际代码修改。Claude生成的方案很好，但执行时需要反复确认。Codex的自动执行能力在这种场景下更合适。

问题的本质是：不同的AI工具有不同的专长，单独使用任何一个都会遇到短板。

---

## 解决方案设计

### 核心理念：分工与协同

我们的方案借鉴了软件工程中的模块化思想。每个AI工具负责自己擅长的领域，通过标准化接口传递中间结果。

Claude担任总架构师，负责任务分解和最终综合。Gemini是外部情报员，专门搜索最新技术资料。Codex是代码工程师，执行具体的审查和修改任务。

这种分工有几个好处。首先，每个工具只处理自己最擅长的任务，输出质量更高。其次，任务可以并行执行，整体耗时缩短。最后，出现问题时容易定位是哪个环节出了差错。

### HANDOFF协议

Multi-CLI协作的核心挑战是上下文传递。一个AI生成的结果如何让另一个AI理解并继续处理？

我们设计了HANDOFF协议来解决这个问题。协议的核心是一个结构化的Markdown文件，记录任务的全部状态。

```markdown
# 🔄 HANDOFF - 项目交接文档

> **创建时间**: 2026-01-10 08:54:26 +0800
> **协议版本**: v2.0
> **当前执行者**: claude
> **任务ID**: multi-cli-20260110

---

## 📋 基础信息

| 项目 | 值 |
|------|-----|
| **项目名称** | myproject |
| **项目路径** | /Users/dev/code/myproject |
| **分析目标** | 深度代码审查 + 架构优化 |
| **预估工时** | 2小时 |

---

## 🎯 任务状态矩阵

| 阶段 | 执行者 | 状态 | 开始时间 | 完成时间 | 关键发现 |
|------|--------|------|----------|----------|----------|
| 1. 项目概览 | Bash | ✅完成 | 08:54:26 | 08:54:30 | 21个活跃项目 |
| 2. 技术调研 | Gemini | ✅完成 | 08:54:35 | 08:56:12 | 发现3个过时依赖 |
| 3. 代码审查 | Codex | ⏳进行中 | 08:56:20 | - | - |
| 4. 架构设计 | Claude | ⏳待开始 | - | - | - |
| 5. 报告生成 | Claude | ⏳待开始 | - | - | - |
```

协议设计遵循几个原则。

**可读性优先**。使用Markdown格式，人类可以直接阅读和编辑。遇到问题时不需要解析复杂的数据结构就能定位原因。

**状态明确**。每个阶段的状态只有三种：待开始、进行中、已完成。状态转换由执行该阶段的工具负责更新。

**上下文结构化**。虽然HANDOFF文件本身是Markdown，但关键数据使用JSON格式嵌入。这样既保证可读性，又便于程序解析。

```json
{
  "stage": "research",
  "executor": "gemini",
  "key_findings": [
    "React 19.2引入了新的并发渲染优化",
    "TypeScript 5.4已修复泛型推断问题"
  ],
  "recommendations": [
    "建议升级React至19.2.1",
    "考虑启用严格模式"
  ],
  "artifacts": [
    "phase2_gemini_research.md",
    "dependency_analysis.json"
  ]
}
```

### 5阶段工作流

完整的Multi-CLI分析包含5个阶段，形成一条流水线。

**Phase 1: 项目概览收集**

这个阶段由Bash脚本执行，不需要AI介入。目的是收集项目的基础信息：目录结构、文件统计、Git历史、README内容等。

```bash
phase1_project_overview() {
    log_step "Phase 1: 收集项目概览..."

    local output_file="$WORKSPACE/phase1_overview.md"

    # 目录结构
    find . -maxdepth 3 -type f | head -100 | sort

    # Git信息
    git log --oneline -10
    git branch -a | head -20

    # 文件统计
    find . -type f -name "*.*" | \
        sed 's/.*\.//' | sort | uniq -c | sort -rn | head -15

    # README内容
    head -100 "README.md"
}
```

这个阶段的输出是后续所有阶段的基础输入。信息收集越完整，后续分析质量越高。

**Phase 2: Gemini技术调研**

Gemini具备联网搜索能力，适合执行技术调研任务。

我们将Phase 1的输出作为上下文，让Gemini搜索相关的最新技术资料。重点关注：使用的技术栈有没有新版本、类似项目采用什么架构模式、有没有已知的安全漏洞。

```bash
phase2_gemini_research() {
    local overview_content=$(head -3000 "$WORKSPACE/phase1_overview.md")

    local prompt="基于以下项目概览，进行技术调研：

$overview_content

请搜索并分析：
1. 该项目使用的技术栈最新版本和最佳实践
2. 类似项目的架构模式
3. 潜在的性能优化方向
4. 安全性考虑

输出Markdown格式的研究报告。"

    timeout $PHASE_TIMEOUT $GEMINI_CMD -p "$prompt" > "$output_file"
}
```

Gemini的输出通常包含大量外部链接和最新信息。这些内容Claude和Codex的知识库中可能没有。

**Phase 3: Codex代码审查**

Codex擅长代码分析和执行。这个阶段我们让它审查项目的实际代码。

代码审查的范围需要控制。不可能把整个代码库都塞进prompt。我们的策略是采样：每种主要语言取5个代表性文件，每个文件取前100行。

```bash
phase3_codex_review() {
    local code_sample=""
    cd "$PROJECT_PATH"

    # 收集主要代码文件
    for pattern in "*.py" "*.ts" "*.js" "*.go" "*.rs"; do
        files=$(find . -name "$pattern" -type f | head -5)
        for f in $files; do
            code_sample+="
### $f
\`\`\`
$(head -100 "$f")
\`\`\`
"
        done
    done

    local prompt="对以下代码进行审查：

$code_sample

请分析：
1. 代码质量问题
2. 潜在的Bug和安全漏洞
3. 性能优化建议
4. 重构机会

输出详细的代码审查报告。"

    timeout $PHASE_TIMEOUT $CODEX_CMD -p "$prompt" > "$output_file"
}
```

采样策略的选择很重要。入口文件、配置文件、核心业务逻辑通常比工具类、测试文件更值得审查。我们的实现还比较粗糙，后续会加入更智能的文件优先级排序。

**Phase 4: Claude架构设计**

前三个阶段收集了足够的信息。现在轮到Claude发挥综合分析能力。

Claude读取所有前序阶段的输出，生成架构改进建议和实施计划。这个阶段的产出是可执行的行动项，而不是泛泛而谈的建议。

```bash
phase4_claude_design() {
    local context=""

    # 收集前序阶段输出
    [ -f "$WORKSPACE/phase1_overview.md" ] && \
        context+="$(cat "$WORKSPACE/phase1_overview.md" | head -2000)\n\n"
    [ -f "$WORKSPACE/phase2_gemini_research.md" ] && \
        context+="$(cat "$WORKSPACE/phase2_gemini_research.md" | head -2000)\n\n"
    [ -f "$WORKSPACE/phase3_codex_review.md" ] && \
        context+="$(cat "$WORKSPACE/phase3_codex_review.md" | head -2000)\n\n"

    local prompt="基于Multi-CLI协作分析的前序阶段输出，进行架构设计：

$context

请生成：
1. 架构改进建议
2. 模块化设计方案
3. 技术债务清理计划
4. 实施优先级排序

输出完整的架构设计文档。"

    timeout $PHASE_TIMEOUT $CLAUDE_CMD -p "$prompt" --no-interactive > "$output_file"
}
```

注意`--no-interactive`参数。自动化流程中不能让Claude等待用户输入。

**Phase 5: 综合报告生成**

最后阶段汇总所有产出，生成一份综合报告。报告包含执行摘要、关键发现、产出文件清单、后续行动项。

这个阶段主要是格式整理，可以用Bash脚本完成，不需要AI介入。

---

## 核心实现

完整的multi-cli-analyze.sh脚本约550行。这里解析几个关键的设计决策。

### CLI可用性检测

不是每个用户都安装了全部三个CLI工具。系统需要优雅地处理这种情况。

```bash
check_cli_availability() {
    log_step "检查CLI可用性..."

    if command -v "$CLAUDE_CMD" &> /dev/null; then
        log_success "Claude CLI: 可用"
        CLAUDE_AVAILABLE=true
    else
        log_warn "Claude CLI: 不可用（跳过Claude阶段）"
        CLAUDE_AVAILABLE=false
    fi

    if command -v "$GEMINI_CMD" &> /dev/null; then
        GEMINI_AVAILABLE=true
    else
        GEMINI_AVAILABLE=false
    fi

    if command -v "$CODEX_CMD" &> /dev/null; then
        CODEX_AVAILABLE=true
    else
        CODEX_AVAILABLE=false
    fi

    # 至少需要一个CLI可用
    if [ "$CLAUDE_AVAILABLE" = false ] && \
       [ "$GEMINI_AVAILABLE" = false ] && \
       [ "$CODEX_AVAILABLE" = false ]; then
        log_error "错误: 没有可用的CLI工具"
        exit 1
    fi
}
```

如果某个CLI不可用，对应的阶段会被跳过。系统继续执行其他可用的阶段。这种设计提高了系统的鲁棒性。

### 超时控制

AI CLI可能因为各种原因卡住。网络问题、模型繁忙、输入过长都可能导致长时间无响应。

我们使用`timeout`命令设置5分钟的硬性时限。

```bash
PHASE_TIMEOUT=300

timeout $PHASE_TIMEOUT $GEMINI_CMD -p "$prompt" > "$output_file" 2>&1 || {
    log_warn "Gemini调研超时或失败"
    echo "# Gemini调研失败" > "$output_file"
    echo "超时或执行错误" >> "$output_file"
}
```

超时后不会中断整个流程。失败会被记录，后续阶段继续执行。最终报告会标注哪些阶段未能完成。

### HANDOFF状态更新

每个阶段完成后需要更新HANDOFF文件中的状态矩阵。我们使用`sed`命令进行原地替换。

```bash
# 更新状态：待开始 → 已完成
sed -i '' 's/| 1\. 项目概览 | Bash | ⏳进行中/| 1. 项目概览 | Bash | ✅完成/' \
    "$WORKSPACE/HANDOFF.md"
```

这种做法有个缺点：依赖文本的精确匹配。如果HANDOFF格式有变化，sed命令可能失效。更稳健的方案是使用专门的Markdown解析器。这是后续优化的方向。

### 工作目录管理

每次分析会创建独立的工作目录，目录名包含项目名和时间戳。

```bash
init_workspace() {
    local project_name="$1"
    WORKSPACE="$ANALYSIS_OUTPUT_DIR/${project_name}_${DATE_TAG}"

    mkdir -p "$WORKSPACE"
}
```

这样设计有几个好处。历史分析结果不会被覆盖。可以方便地对比不同时间的分析结果。清理旧数据也很简单，直接删除目录即可。

---

## 实战案例

我们用这套系统分析了`novel-ai-generator`项目。这是一个AI小说生成器，包含API服务、Gradio界面、n8n工作流。

### 执行过程

启动命令：

```bash
~/.claude/scripts/multi-cli-analyze.sh \
    ~/Documents/code/novel-ai-generator-original \
    --full
```

输出日志：

```
==============================================
       Multi-CLI 项目协作分析系统
==============================================

[INFO] 项目: novel-ai-generator-original
[INFO] 路径: /Users/anwu/Documents/code/novel-ai-generator-original

[STEP] 检查CLI可用性...
[✓] Claude CLI: 可用
[!] Gemini CLI: 不可用（跳过Gemini阶段）
[!] Codex CLI: 不可用（跳过Codex阶段）

[STEP] 初始化工作目录: /Users/anwu/analysis_results/novel-ai-generator-original_20260110_084521
[✓] HANDOFF文件已创建

[STEP] Phase 1: 收集项目概览...
[✓] Phase 1完成

[STEP] Phase 4: Claude架构设计...
[✓] Phase 4完成

[STEP] Phase 5: 生成综合报告...
[✓] Phase 5完成

==============================================
[✓] Multi-CLI分析完成！
==============================================
```

这次测试只有Claude可用，Gemini和Codex被跳过。系统仍然完成了可执行的阶段并生成了报告。

### 分析结果

Phase 1收集到的项目信息：

- 文件总数：156个
- 主要语言：Python（89个文件）
- 依赖管理：requirements.txt
- 测试框架：pytest
- 最近活跃：5个提交在24小时内

Claude的架构建议（摘录）：

```markdown
## 架构改进建议

### 1. 模块化重构
当前core目录下的agent_coordinator.py和novel_orchestrator.py职责重叠。
建议合并为统一的编排层。

### 2. 配置管理
发现硬编码配置散落在多个文件中。
建议引入pydantic-settings统一管理。

### 3. 测试覆盖
tests目录已有基础框架，但覆盖率低于30%。
优先补充core模块的单元测试。

## 实施优先级

| 优先级 | 任务 | 预估工时 | 依赖 |
|--------|------|----------|------|
| P0 | 配置管理统一 | 2h | 无 |
| P1 | 核心模块测试 | 4h | P0 |
| P2 | 编排层重构 | 6h | P1 |
```

这些建议直接可执行。每个任务都有明确的范围、工时估算和依赖关系。

---

## 最佳实践

### 应该做的

**每次阶段完成后立即更新HANDOFF**。状态延迟更新会导致混乱，尤其是出现异常需要排查时。

**使用结构化JSON传递复杂上下文**。虽然Markdown可读性好，但复杂数据用JSON更可靠。两者结合使用效果最佳。

**保留所有中间产出文件**。即使某个阶段失败了，之前的产出仍然有价值。不要因为最终报告没生成就删除中间文件。

**记录执行时间**。对于优化分析流程很有帮助。哪个阶段耗时最长？有没有可以并行的阶段？

### 不应该做的

**不要在不同CLI之间直接传递大段文本**。通过HANDOFF文件和中间文件传递。大文本容易被截断或产生解析错误。

**不要跳过HANDOFF状态更新**。哪怕只是做测试。养成习惯后遇到真正的问题才不会忘记。

**不要覆盖其他CLI的产出文件**。每个阶段写自己的文件。需要修改前序产出时，创建新文件记录修改版本。

**不要假设上一阶段的结果**。即使你知道Gemini会输出什么格式，也要处理它可能失败或输出异常的情况。

---

## 扩展方向

当前实现是MVP版本，有几个明显的改进方向。

**并行执行**。目前5个阶段是串行的。实际上Phase 2（Gemini调研）和Phase 3（Codex审查）没有依赖关系，可以并行。

**智能文件选择**。Phase 3的代码采样策略太简单。可以引入代码复杂度分析，优先选择复杂度高、变更频繁的文件。

**增量分析**。对于之前分析过的项目，可以只分析变更的部分。基于Git diff识别变更范围。

**结果缓存**。技术调研的结果有一定时效性，但不需要每次都重新搜索。可以缓存24小时内的调研结果。

---

## 部署指南

### 前置条件

至少安装以下CLI工具之一：

```bash
# Claude Code CLI（必选）
npm install -g @anthropic/claude-code

# Gemini CLI（可选，用于联网调研）
pip install google-generativeai

# Codex CLI（可选，用于代码审查）
npm install -g @openai/codex
```

### 安装脚本

```bash
# 克隆配置
mkdir -p ~/.claude/scripts
curl -o ~/.claude/scripts/multi-cli-analyze.sh \
    https://raw.githubusercontent.com/.../multi-cli-analyze.sh
chmod +x ~/.claude/scripts/multi-cli-analyze.sh

# 创建输出目录
mkdir -p ~/analysis_results

# 测试运行
~/.claude/scripts/multi-cli-analyze.sh /path/to/project
```

### 常用命令

```bash
# 完整分析
~/.claude/scripts/multi-cli-analyze.sh /path/to/project --full

# 仅技术调研
~/.claude/scripts/multi-cli-analyze.sh /path/to/project --research

# 仅代码审查
~/.claude/scripts/multi-cli-analyze.sh /path/to/project --review

# 仅架构设计
~/.claude/scripts/multi-cli-analyze.sh /path/to/project --design
```

---

## 经验总结

经过一个月的使用，我们总结了几点经验。

**工具选择很重要**。不是所有任务都需要Multi-CLI。简单的代码修改用单个Claude足够了。只有需要联网搜索、大规模代码审查、或者需要不同视角的任务才值得启动完整流程。

**协议设计比实现重要**。HANDOFF协议定义了系统的骨架。一旦协议稳定，具体实现可以随时替换。我们已经迭代了3个版本的Bash脚本，但协议只有2个版本。

**容错是必须的**。AI CLI的行为不是100%可预测的。相同的输入可能得到不同质量的输出。超时、失败、异常格式都要处理。

**人工审核不可省略**。这套系统的产出是建议和计划，不是最终决策。我们从不让系统自动执行代码修改。所有变更都需要人工审核后执行。

Multi-CLI协作框架仍在演进中。随着更多AI工具的出现，协作的模式也会变化。但核心理念不变：发挥各工具专长，通过标准化接口协同工作。

---

## 参考资料

1. oh-my-opencode - Sisyphus Agent Pattern
2. MetaGPT - Multi-Agent Framework
3. HANDOFF Protocol Specification v2.0
4. Claude Code CLI Documentation
5. Gemini CLI Best Practices

---

**关于作者**

Innora Security Research Team专注于AI安全与自动化研究。我们定期分享工程实践和技术探索成果。

---

*本文基于Multi-CLI协作系统的实际开发和使用经验撰写。所有代码示例均来自生产环境。*

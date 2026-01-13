# 夜间自主AI编排系统：让AI在你睡觉时工作

> **作者**: Innora Security Research Team
> **发布日期**: 2026年1月10日
> **阅读时长**: 约15分钟
> **技术难度**: 高级

---

## 执行摘要

每天凌晨1点到6点，我的电脑自动运行一套AI分析系统。它收集我前一天的操作日志，分析代码变更，调用多个AI工具进行深度审查，最后生成一份报告等我起床查看。

这套系统我称之为Nighttime Orchestrator。本文完整介绍它的设计思路、架构实现和部署经验。你将获得可直接使用的Bash脚本、Python分析器和macOS launchd配置。

核心价值：把低价值的重复性分析任务自动化，让AI在闲置时段完成，节省宝贵的工作时间。

---

## 问题背景

我使用Claude Code已经超过3个月。期间积累了大量操作数据：执行过哪些Bash命令、修改过哪些文件、在哪些项目上花了最多时间。

这些数据散落在日志文件里，从未被利用。偶尔我会手动翻看，发现自己重复执行过很多相似的命令。要是有个工具能自动分析这些模式，提取可自动化的任务就好了。

另一个痛点是代码审查。我的工作目录有20多个活跃项目。每次切换项目都要回忆之前做到哪里了。如果能有个系统每天自动扫描所有项目的变更，生成一份概览就好了。

最后是多AI协作的需求。Claude擅长深度分析，Gemini可以联网搜索最新资料，Codex能自动执行代码修改。但手动协调它们太麻烦了。

把这三个需求结合起来，就诞生了Nighttime Orchestrator。

---

## 设计目标

在动手写代码之前，我明确了几个设计目标。

**无人值守运行**。系统必须能在我睡觉时自动执行，不需要任何人工干预。出错时优雅降级而不是崩溃。

**资源友好**。夜间运行不意味着可以随便消耗资源。系统应该使用低优先级，不影响其他后台任务。

**结果可追溯**。每次运行产生的日志、分析结果、报告都要保留。方便回溯排查问题。

**渐进式改进**。第一版不求完美。能跑起来比什么都重要。后续根据实际使用情况迭代。

---

## 系统架构

整个系统分为三层：调度层、编排层、执行层。

```
┌─────────────────────────────────────────────────────────┐
│                     调度层 (launchd)                     │
│  - 时间触发: 每天 1:30 AM                                │
│  - 资源控制: Nice=10, LowPriorityIO                     │
│  - 超时保护: 最长4小时                                   │
├─────────────────────────────────────────────────────────┤
│                     编排层 (Orchestrator)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Phase 1  │→ │ Phase 2  │→ │ Phase 3  │→ ...         │
│  │ 日志收集 │  │ 代码分析 │  │ AI协调   │              │
│  └──────────┘  └──────────┘  └──────────┘              │
├─────────────────────────────────────────────────────────┤
│                     执行层 (Multi-CLI)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │  Claude  │  │  Codex   │  │  Gemini  │  │ Ollama  ││
│  │ 主分析   │  │ 代码改进 │  │ 联网调研 │  │ 本地推理││
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘│
└─────────────────────────────────────────────────────────┘
```

### 调度层

macOS的launchd负责定时触发。我选择凌晨1:30作为启动时间，避开整点可能的系统维护任务。

launchd配置文件位于`~/Library/LaunchAgents/com.anwu.nighttime-ai.plist`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.anwu.nighttime-ai</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/anwu/.claude/scripts/nighttime-ai-orchestrator.sh</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>1</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>

    <key>Nice</key>
    <integer>10</integer>

    <key>LowPriorityIO</key>
    <true/>

    <key>ProcessType</key>
    <string>Background</string>

    <key>StandardOutPath</key>
    <string>/Users/anwu/.claude/nighttime/logs/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/anwu/.claude/nighttime/logs/launchd_stderr.log</string>
</dict>
</plist>
```

几个关键配置值得说明。`Nice`设为10表示降低进程优先级。`LowPriorityIO`设为true表示降低磁盘IO优先级。`ProcessType`设为Background告诉系统这是后台任务。

这些设置确保夜间任务不会影响系统的正常运行。即使我凌晨还在用电脑，也不会感到卡顿。

### 编排层

核心是一个Bash脚本：`nighttime-ai-orchestrator.sh`。脚本组织成6个阶段，顺序执行。

```bash
#!/bin/bash
set -e

# 全局配置
export NIGHTTIME_HOME="$HOME/.claude/nighttime"
export LOG_DIR="$NIGHTTIME_HOME/logs"
export ANALYSIS_DIR="$NIGHTTIME_HOME/analysis"
export REPORTS_DIR="$NIGHTTIME_HOME/reports"
export STATE_FILE="$NIGHTTIME_HOME/state.json"

# 时间窗口配置
START_HOUR=1
END_HOUR=6

# 日期标记
DATE_TAG=$(date '+%Y%m%d')
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %z')
```

时间窗口检查是第一个防护措施。如果不在1-6点之间，脚本直接退出。这防止了误触发。

```bash
check_time_window() {
    local current_hour=$(date '+%H')
    current_hour=$((10#$current_hour))  # 移除前导零

    if [ $current_hour -ge $START_HOUR ] && [ $current_hour -lt $END_HOUR ]; then
        return 0  # 在时间窗口内
    else
        return 1  # 不在时间窗口内
    fi
}
```

当然也提供了`--force`参数用于手动测试。

### 执行层

执行层是多个AI CLI工具的集合。每个工具有明确的职责。

| 工具 | 职责 | 调用方式 |
|------|------|----------|
| Claude | 深度代码分析、报告生成 | `claude -p "..." --no-interactive` |
| Codex | 自动代码修改、重构执行 | `codex -p "..." --auto-edit` |
| Gemini | 联网技术调研、版本检查 | `gemini -p "..."` |
| Ollama | 本地敏感代码分析 | `ollama run qwen3:14b` |

执行层的一个重要设计是容错。任何一个工具不可用时，对应阶段会被跳过，不影响其他阶段。

---

## 6阶段工作流

### Phase 1: 操作日志收集

第一阶段扫描Claude Code的日志目录，收集过去24小时的操作记录。

```bash
collect_operation_logs() {
    log_info "Phase 1: 收集Claude Code操作日志..."

    local output_file="$ANALYSIS_DIR/operations_$DATE_TAG.json"

    # 日志源
    local bash_log="$CLAUDE_LOGS_DIR/bash-commands.log"
    local file_log="$CLAUDE_LOGS_DIR/file-changes.log"
    local session_log="$CLAUDE_LOGS_DIR/sessions.log"

    cat > "$output_file" << EOF
{
  "collection_time": "$TIMESTAMP",
  "date": "$DATE_TAG",
  "sources": {
    "bash_commands": {
      "path": "$bash_log",
      "exists": $([ -f "$bash_log" ] && echo "true" || echo "false"),
      "size_bytes": $([ -f "$bash_log" ] && stat -f%z "$bash_log" 2>/dev/null || echo "0")
    },
    "file_changes": {
      "path": "$file_log",
      "exists": $([ -f "$file_log" ] && echo "true" || echo "false")
    }
  }
}
EOF

    if [ -f "$bash_log" ]; then
        local cmd_count=$(wc -l < "$bash_log" | tr -d ' ')
        log_info "  - Bash命令日志: $cmd_count 条记录"
    fi
}
```

输出是一个JSON文件，记录日志源的元信息。后续Python分析器会进一步处理这些数据。

### Phase 2: 代码变更分析

第二阶段扫描代码目录，找出72小时内有更新的项目。

```bash
analyze_code_changes() {
    log_info "Phase 2: 分析代码目录变更..."

    local output_file="$ANALYSIS_DIR/code_changes_$DATE_TAG.json"
    local projects_file="$ANALYSIS_DIR/active_projects_$DATE_TAG.txt"

    # 查找72小时内有更新的项目
    find "$CODE_DIR" -maxdepth 3 -name ".git" -type d -mtime -3 2>/dev/null | \
        while read git_dir; do
            dirname "$git_dir"
        done > "$projects_file"

    local project_count=$(wc -l < "$projects_file" | tr -d ' ')
    log_info "  - 发现 $project_count 个活跃项目（72小时内有更新）"

    cat > "$output_file" << EOF
{
  "analysis_time": "$TIMESTAMP",
  "code_directory": "$CODE_DIR",
  "active_projects_count": $project_count,
  "time_window_hours": 72,
  "projects_file": "$projects_file"
}
EOF
}
```

用`.git`目录作为项目识别标志。扫描深度限制在3层，避免进入node_modules这类大目录。

### Phase 3: Multi-CLI协调分析

第三阶段是核心。它创建HANDOFF文件，协调多个AI工具进行分析。

```bash
multi_cli_analysis() {
    log_info "Phase 3: Multi-CLI协调分析..."

    local operations_file="$1"
    local changes_file="$2"
    local handoff_file="$ANALYSIS_DIR/HANDOFF_$DATE_TAG.md"

    # 创建HANDOFF文件
    cat > "$handoff_file" << EOF
# 🔄 HANDOFF - 夜间自动分析交接文档

> **创建时间**: $TIMESTAMP
> **协议版本**: v2.0
> **当前执行者**: claude
> **任务ID**: nighttime-$DATE_TAG

---

## 🎯 任务状态矩阵

| 阶段 | 执行者 | 状态 | 开始时间 | 完成时间 | 关键发现 |
|------|--------|------|----------|----------|----------|
| 1. 操作日志收集 | Bash | ✅完成 | $TIMESTAMP | $TIMESTAMP | 日志已收集 |
| 2. 代码变更分析 | Bash | ✅完成 | $TIMESTAMP | $TIMESTAMP | 活跃项目已识别 |
| 3. 深度代码审查 | Claude | ⏳进行中 | - | - | - |
| 4. 优化建议生成 | Gemini | ⏳待开始 | - | - | - |
| 5. 代码自动改进 | Codex | ⏳待开始 | - | - | - |
EOF
```

接下来检测各CLI的可用性，调用可用的工具执行分析。

```bash
    # 检查CLI可用性
    if command -v "$CLAUDE_CMD" &> /dev/null; then
        log_info "  - 启动Claude深度分析..."

        local claude_prompt="基于以下夜间自动分析上下文，执行深度代码审查：

HANDOFF文件: $handoff_file
操作日志: $operations_file
代码变更: $changes_file

请分析：
1. 用户操作模式中的重复工作
2. 代码质量问题和改进机会
3. 可自动化优化的任务
4. 生成优化建议报告"

        local claude_output="$REPORTS_DIR/claude_analysis_$DATE_TAG.md"

        # 非交互模式执行，带超时
        timeout 300 $CLAUDE_CMD -p "$claude_prompt" --no-interactive > "$claude_output" 2>&1 || {
            log_warn "  - Claude分析超时或失败，继续执行..."
        }
    fi
```

5分钟超时设置很重要。没有超时的话，CLI卡住会导致整个流程阻塞。

### Phase 4: 报告生成

第四阶段汇总所有信息，生成每日报告。

```bash
generate_daily_report() {
    log_info "Phase 4: 生成每日夜间分析报告..."

    local report_file="$REPORTS_DIR/nighttime_report_$DATE_TAG.md"

    cat > "$report_file" << EOF
# 🌙 夜间自主AI分析报告

> **生成时间**: $TIMESTAMP
> **分析周期**: $(date -v-1d '+%Y-%m-%d') 00:00 ~ $(date '+%Y-%m-%d') 06:00
> **执行系统**: Nighttime AI Orchestrator v1.0.0

---

## 📊 执行摘要

| 指标 | 值 |
|------|-----|
| **开始时间** | $TIMESTAMP |
| **执行阶段** | 4/4 |
| **分析状态** | ✅ 完成 |

---

## 📁 生成文件

| 文件 | 路径 | 用途 |
|------|------|------|
| 操作日志 | $ANALYSIS_DIR/operations_$DATE_TAG.json | Claude Code操作记录 |
| 代码变更 | $ANALYSIS_DIR/code_changes_$DATE_TAG.json | 72小时代码变更 |
| HANDOFF | $ANALYSIS_DIR/HANDOFF_$DATE_TAG.md | Multi-CLI协作文档 |

---

**下次执行时间**: $(date -v+1d '+%Y-%m-%d') 01:00
EOF
}
```

报告格式固定，方便后续自动化处理或发送通知。

### Phase 5: 自我迭代循环

这个阶段调用自我迭代引擎，分析操作模式并生成改进建议。

```bash
run_self_iteration() {
    log_info "Phase 5: 执行自我迭代循环..."

    local iteration_script="$HOME/.claude/scripts/self-iteration-engine.py"

    if [ ! -f "$iteration_script" ]; then
        log_warn "  - 自我迭代引擎未找到"
        return
    fi

    # 首先运行Python分析器
    local analyzer_script="$HOME/.claude/scripts/analyze-operations.py"
    if [ -f "$analyzer_script" ]; then
        python3 "$analyzer_script" --hours 72 --format json > /dev/null 2>&1 || {
            log_warn "  - 操作分析失败，跳过自我迭代"
            return
        }
    fi

    # 运行自我迭代引擎（dry-run模式）
    python3 "$iteration_script" --dry-run > /dev/null 2>&1 && {
        log_info "  - 自我迭代完成"
    }
}
```

dry-run模式意味着只生成建议，不自动应用。所有变更都需要人工审核。

### Phase 6: 状态更新

最后阶段更新运行状态，记录统计信息。

```bash
update_state() {
    log_info "Phase 6: 更新系统状态..."

    local total_runs=$(jq -r '.total_runs // 0' "$STATE_FILE" 2>/dev/null || echo "0")
    total_runs=$((total_runs + 1))

    cat > "$STATE_FILE" << EOF
{
  "version": "1.0.0",
  "last_run": "$TIMESTAMP",
  "total_runs": $total_runs,
  "improvements_made": 0,
  "status": "completed",
  "last_report": "$REPORTS_DIR/nighttime_report_$DATE_TAG.md"
}
EOF
}
```

状态文件用于跟踪系统运行历史。累计运行次数、最后运行时间、已应用的改进数量都记录在这里。

---

## 目录结构

系统的所有文件都集中在`~/.claude/nighttime/`目录下：

```
~/.claude/nighttime/
├── analysis/                      # 分析输出
│   ├── operations_YYYYMMDD.json   # 操作日志
│   ├── code_changes_YYYYMMDD.json # 代码变更
│   ├── HANDOFF_YYYYMMDD.md        # 交接文档
│   └── active_projects_*.txt      # 活跃项目列表
├── iterations/                    # 迭代记录
│   └── iteration_report_*.md      # 迭代报告
├── logs/                          # 日志目录
│   ├── orchestrator_YYYYMMDD.log  # 编排器日志
│   ├── launchd_stdout.log         # launchd标准输出
│   └── launchd_stderr.log         # launchd标准错误
├── reports/                       # 报告输出
│   ├── nighttime_report_*.md      # 每日报告
│   └── claude_analysis_*.md       # Claude分析
└── state.json                     # 运行状态
```

这种目录结构有几个好处。按日期命名可以轻松找到历史文件。类型分目录便于批量清理。单一根目录方便备份。

---

## 实际运行数据

系统上线一周后，我收集了一些运行数据。

### 执行时间统计

| 阶段 | 平均耗时 | 最长耗时 |
|------|----------|----------|
| Phase 1: 日志收集 | 2秒 | 5秒 |
| Phase 2: 代码分析 | 3秒 | 8秒 |
| Phase 3: Multi-CLI | 45秒 | 180秒 |
| Phase 4: 报告生成 | 1秒 | 2秒 |
| Phase 5: 自我迭代 | 5秒 | 15秒 |
| Phase 6: 状态更新 | 1秒 | 1秒 |
| **总计** | **~1分钟** | **~3分钟** |

Phase 3耗时最长，主要取决于Claude分析的复杂度。如果分析的项目多、代码量大，时间会更长。

### 发现的操作模式

第一周分析了7125条Bash命令。高频命令统计：

| 命令模式 | 执行次数 | 自动化潜力 |
|----------|----------|------------|
| `git status` | 312 | 低（信息查看） |
| `git add .` | 287 | 中（可脚本化） |
| `npm install` | 156 | 低（依赖安装） |
| `python3 xxx.py` | 134 | 高（可封装） |
| `docker-compose up` | 98 | 高（可自动化） |

基于这些数据，系统生成了2个自动化建议。一个是创建`Bas`别名（alias），另一个是生成常用命令的自动化脚本。

### 资源消耗

| 指标 | 值 |
|------|-----|
| CPU平均占用 | 15%（Nice=10生效） |
| 内存峰值 | 200MB |
| 磁盘写入 | ~5MB/次 |
| 网络请求 | 取决于CLI调用 |

资源消耗在可接受范围内。没有观察到对其他后台任务的明显影响。

---

## 服务管理

系统提供了一个管理脚本`automation-manager.sh`，统一管理所有launchd服务。

### 查看状态

```bash
~/.claude/scripts/automation-manager.sh status
```

输出示例：
```
=== 自动化服务状态 ===

com.anwu.nighttime-ai
  状态: 已加载
  PID: -
  最后执行: 2026-01-10 01:30:00
  下次执行: 2026-01-11 01:30:00

com.anwu.daily-plan
  状态: 已加载
  PID: -
  最后执行: 2026-01-10 07:00:00
  下次执行: 2026-01-11 07:00:00
```

### 手动触发

```bash
# 强制执行夜间编排（跳过时间窗口）
~/.claude/scripts/nighttime-ai-orchestrator.sh --force

# 或通过launchctl
launchctl start com.anwu.nighttime-ai
```

### 查看日志

```bash
# 编排器日志
tail -f ~/.claude/nighttime/logs/orchestrator_$(date +%Y%m%d).log

# launchd日志
tail -f ~/.claude/nighttime/logs/launchd_stdout.log
```

---

## 故障排查

### 服务未运行

```bash
# 检查launchd状态
launchctl list | grep nighttime-ai

# 如果未加载，重新加载
launchctl load ~/Library/LaunchAgents/com.anwu.nighttime-ai.plist
```

### 脚本执行失败

```bash
# 检查权限
ls -la ~/.claude/scripts/nighttime-ai-orchestrator.sh

# 赋予执行权限
chmod +x ~/.claude/scripts/nighttime-ai-orchestrator.sh

# 手动测试
~/.claude/scripts/nighttime-ai-orchestrator.sh --force
```

### CLI不可用

```bash
# 检查CLI安装
which claude gemini codex

# 检查PATH
echo $PATH
```

如果CLI在交互式shell可用但launchd中不可用，可能是PATH问题。在脚本开头添加：

```bash
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
```

---

## 安全考虑

夜间自动运行的系统需要额外的安全考量。

**权限最小化**。所有脚本都在用户空间运行，不需要root权限。不访问系统目录。

**超时保护**。每个阶段都有超时设置。整体运行时间不会超过4小时。

**资源限制**。通过launchd的Nice和LowPriorityIO限制资源占用。

**日志审计**。所有操作都有完整日志。定期检查日志可以发现异常行为。

**不自动修改代码**。自我迭代引擎默认使用dry-run模式。所有代码变更建议需要人工审核后执行。

---

## 演进规划

当前版本是v1.0.0，功能比较基础。规划的后续改进：

**短期（1周内）**
- 监控服务运行稳定性
- 收集更多操作模式数据
- 修复发现的边界情况

**中期（1月内）**
- 优化自我迭代的置信度阈值
- 添加更多改进模板
- 支持并行执行部分阶段

**长期（3月内）**
- 实现自动代码改进（需人工审核流程）
- 集成Notion同步
- 添加邮件/Slack通知

---

## 经验总结

这套系统运行一周后，我总结了几点经验。

**简单优于复杂**。第一版只有4个阶段，后来才扩展到6个。过早优化会增加维护负担。

**容错是核心**。任何一个环节失败都不应该导致整个系统崩溃。超时、跳过、降级是关键设计。

**日志要详细**。夜间运行意味着出问题时你在睡觉。详细的日志是排查问题的唯一依靠。

**人工审核不可省**。AI生成的建议可能有错。特别是代码修改，必须人工确认后执行。

**渐进式改进**。不要期望一次做对所有事情。上线、观察、迭代是正确的节奏。

Nighttime Orchestrator让我每天早上起床就能看到一份分析报告。知道昨天花了多少时间在哪些项目上，有哪些可以优化的操作模式，哪些代码需要关注。这些信息在以前需要我手动整理，现在AI自动完成。

节省的时间用来做更有价值的事情。这就是自动化的意义。

---

## 参考资料

1. Apple launchd.plist Documentation
2. Agents 2.0 - Self-Evolving AI Architecture
3. oh-my-opencode - Sisyphus Agent Pattern
4. HANDOFF Protocol Specification v2.0
5. Claude Code CLI Reference

---

**关于作者**

Innora Security Research Team专注于AI安全与自动化研究。我们相信好的工具应该在你不注意的时候默默工作。

---

*本文基于Nighttime Orchestrator v1.0.0的实际部署经验撰写。系统已稳定运行超过一周。*

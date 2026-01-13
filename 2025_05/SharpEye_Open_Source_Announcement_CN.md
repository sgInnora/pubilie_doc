# SharpEye: 开源高级Linux入侵检测系统

![SharpEye 标志](https://github.com/sgInnora/sharpeye/raw/main/assets/logo.png)

## SharpEye简介: Linux安全领域的新纪元

我们非常兴奋地宣布，我们的高级Linux入侵检测与威胁狩猎系统 **SharpEye** 现已开源。经过多年的开发和完善，innora.ai团队决定将这款强大的安全工具与全球社区共享，以提升全球Linux环境的整体安全状况。

**GitHub仓库**: [https://github.com/sgInnora/sharpeye](https://github.com/sgInnora/sharpeye)

### 为什么我们创建SharpEye

在当今日益复杂的威胁环境中，传统安全工具往往无法检测到复杂的攻击。现代攻击者采用先进技术，可以绕过常规检测机制，即使系统部署了安全措施，依然存在着安全隐患。

SharpEye的设计初衷就是应对这些挑战，它结合了传统的系统监控与前沿的机器学习和行为分析技术。我们的目标是创建一个安全解决方案，能够：

1. 检测异常行为，而不仅仅是匹配已知特征
2. 提供对Linux系统各个方面的全面可视性
3. 适应不断演变的威胁，无需持续的手动更新
4. 为各种规模的组织提供可访问的安全工具

最终的结果就是SharpEye：一个强大、智能且适应性强的安全监控框架，可以深入洞察Linux系统，识别其他工具可能会忽略的可疑活动。

## 全面的功能集

SharpEye通过对Linux系统各个方面的监控，构建了全面的安全视图，其独特之处在于多方位的系统安全方法：

### 基于机器学习的系统资源监控

SharpEye持续分析CPU、内存和磁盘使用模式，以识别可能表明存在加密货币挖矿、资源劫持或拒绝服务攻击的异常资源消耗。我们增强的系统资源模块结合了传统的基于阈值的监控和先进的机器学习功能：

- 使用隔离森林算法检测异常资源模式
- 识别超出正常操作模式的异常CPU或内存使用峰值
- 检测具有异常稳定性指标的持续高资源使用
- 分析可能表明数据泄露的意外磁盘I/O模式
- 执行跨资源相关性分析以检测复杂的攻击模式
- 进行时间序列分析以识别令人担忧的趋势
- 生成正常行为的统计模型进行比较
- 实现自我训练能力以适应您的环境
- 监控隐藏进程和可疑执行模式
- 提供资源使用异常的详细指标和严重性评级

### 用户账户安全

攻击者经常将用户账户作为入口点或权限提升的目标。SharpEye提供了全面的用户账户活动监控：

- 检测未授权的账户创建或修改
- 监控权限提升尝试
- 分析可疑登录模式，包括异常登录时间或位置
- 识别突然显示活动的休眠账户
- 检测异常的凭证使用或密码更改

### 基于机器学习的进程分析

SharpEye的核心是一个高级进程监控系统，采用基于规则和机器学习的方法：

- 进程谱系分析，识别可疑的父子关系
- 行为分析，检测表现出异常模式的进程
- 识别试图隐藏其活动的进程
- 内存扫描，寻找代码注入或篡改的迹象
- 检测已知的恶意进程签名及其变种
- 全面的进程关系映射，识别攻击链
- 实时进程执行监控与异常检测
- 进程环境和参数分析，识别可疑配置
- 与网络和文件活动的交叉关联，构建整体威胁模型
- 进程完整性验证，检测运行时修改

### 基于机器学习的加密货币挖矿检测

未授权的加密货币挖矿是最常见的资源盗窃形式之一。SharpEye包含专门针对检测加密货币挖矿而训练的机器学习算法：

- CPU使用模式分析，识别挖矿算法
- 检测挖矿流量和通信模式
- 识别挖矿软件特征和行为
- 进程行为与已知挖矿模式的统计分析对比
- 多指标关联，实现高可信度检测

### 网络连接监控

SharpEye仔细监控网络连接，提供以下可视性：

- 连接到潜在恶意域名或IP的异常出站连接
- 检测数据泄露模式
- 识别命令与控制(C2)流量
- 监控意外的监听端口或服务
- 分析加密流量模式的异常

### 威胁情报集成

SharpEye连接多个威胁情报源，以增强检测能力：

- 根据已知恶意IP数据库验证网络连接
- 根据威胁情报源检查文件哈希
- 定期更新威胁签名和妥协指标
- 可定制的威胁源集成，适应特定环境
- 将本地发现与全球威胁形势关联

### 文件系统完整性

维护文件系统完整性对安全至关重要：

- 使用加密哈希验证系统文件完整性
- 检测对关键系统文件的未授权更改
- 监控敏感配置文件
- 识别隐藏文件或异常权限更改
- 检测潜在的后门或被特洛伊化的二进制文件

### 日志分析引擎

SharpEye包含一个复杂的日志分析引擎，可以：

- 监控并分析系统日志中的可疑模式
- 关联不同服务的日志事件
- 检测日志篡改或删除尝试
- 识别身份验证失败和暴力尝试
- 提供日志异常的上下文感知分析

### 计划任务检查

攻击者经常使用计划任务来维持持久性：

- 识别可疑的cron作业和计划任务
- 检测对现有计划任务的修改
- 监控异常的调度模式
- 分析计划脚本的内容和行为
- 将计划任务与其他系统活动关联

### SSH安全与高级分析

SSH是攻击者常见的目标，我们全面的SSH分析器提供行业领先的检测能力：

- 监控SSH配置中的安全弱点和最佳实践
- 检测未授权访问尝试和暴力攻击
- 识别异常的SSH客户端配置或密钥更改
- 监控SSH会话活动的异常行为
- 高级SSH隧道和端口转发检测
- SSH密钥使用模式分析和异常检测
- 检测从多个位置或在异常时间使用的密钥
- SSH配置的基线创建和比较
- SSH密钥自动化分析（cron作业、systemd服务）
- 与威胁情报集成，关联可疑访问

### 内核模块分析

Linux内核是高价值目标：

- 检测恶意内核模块和rootkit
- 验证内核模块签名和完整性
- 监控内核模块的加载和卸载
- 识别内核内存篡改的迹象
- 分析内核级行为异常

### 库检查

动态库经常被利用：

- 识别库劫持尝试
- 检测预加载攻击(LD_PRELOAD)
- 验证库完整性和签名
- 监控敏感进程中的库加载
- 检测异常的库依赖关系

### 权限提升检测

SharpEye积极寻找权限提升的迹象：

- 监控可利用的错误配置
- 检测SUID/SGID二进制文件的利用
- 识别正在使用的潜在内核漏洞
- 监控对易受攻击服务的利用
- 检测可疑的能力变化

### 高级Rootkit检测

我们全面的Rootkit检测模块提供了最先进的功能：

- 内核级完整性验证，用于检测低级修改
- 通过多种方法检测隐藏进程（系统调用钩子、/proc比较）
- 检测被劫持的中断处理程序和服务例程
- 网络堆栈完整性验证，识别隐蔽通道
- 隐藏文件系统对象检测（包括覆盖技术）
- 基于内存的rootkit检测（无持久组件）
- 通过多种查询方法对系统信息进行交叉验证
- 对意外内核行为的异常检测
- 运行时内核完整性测量和验证
- 行为分析，检测rootkit规避技术

## SharpEye开源的意义

将SharpEye开源代表了我们对几个核心原则的承诺：

### 安全民主化

通过开源SharpEye，我们旨在为各种规模和资源的组织提供先进的安全能力。安全不应该是资金充足实体的特权，而应该是所有人的权利。SharpEye为所有人提供企业级安全监控，从个人Linux爱好者到大型组织。

### 透明度与信任

安全工具必须值得信任。通过将我们的代码开放给公众审查，我们展示了对透明度的承诺。用户可以确切地验证SharpEye的运行方式，识别潜在的隐私问题，并确认工具按预期运行。

### 协作创新

安全是一个不断发展的挑战，从不同角度的贡献中获益。通过邀请全球社区为SharpEye做出贡献，我们正在实现集体创新，远超任何单一组织所能实现的成果。一起，我们可以更有效地应对新兴威胁并开发新的检测能力。

### 知识共享

当知识被共享时，安全社区会变得更强大。SharpEye不仅提供工具，还作为一个教育平台，用于理解Linux安全、入侵检测技术和机器学习在网络安全中的应用。学生、研究人员和专业人士都可以从我们的工作中学习并在此基础上发展。

## SharpEye入门指南

### 系统要求

SharpEye设计为在大多数Linux发行版上高效运行，占用资源最小化：

- **操作系统**：基于Linux的操作系统（Debian、Ubuntu、CentOS、RHEL等）
- **Python**：Python 3.6或更高版本
- **权限**：需要root权限进行全面扫描
- **磁盘空间**：最小化（安装约50MB，日志存储可变）
- **内存**：至少512MB RAM（推荐1GB+）

### 安装

SharpEye的安装非常简单：

```bash
# 克隆仓库
git clone https://github.com/sgInnora/sharpeye.git

# 切换到SharpEye目录
cd sharpeye

# 运行安装脚本
sudo ./install.sh
```

安装脚本将：
- 安装必要的依赖
- 设置必要的目录
- 配置基本设置
- 安装SharpEye系统服务
- 创建计划扫描

### 基本用法

SharpEye提供几种操作模式以适应不同的安全需求：

#### 全系统扫描

要对系统进行全面扫描：

```bash
sudo sharpeye --full-scan
```

这将激活所有检测模块并生成潜在安全问题的详细报告。

#### 目标模块扫描

要运行特定的检测模块：

```bash
sudo sharpeye --module network
sudo sharpeye --module cryptominer
sudo sharpeye --module kernel
```

#### 基线比较

SharpEye可以建立"正常"系统行为的基线，并将未来扫描与之比较：

```bash
# 当系统处于已知良好状态时建立基线
sudo sharpeye --establish-baseline

# 将当前状态与基线进行比较
sudo sharpeye --compare-baseline
```

这种方法特别有效，可用于检测可能表明系统被入侵的细微变化。

#### 持续监控

为了持续保护，SharpEye可以配置为服务运行：

```bash
# 启动SharpEye服务
sudo systemctl start sharpeye

# 启用SharpEye在引导时启动
sudo systemctl enable sharpeye
```

### 配置

SharpEye高度可配置，可以适应不同的环境：

- 配置文件存储在`/etc/sharpeye/`
- 主配置文件是`config.yaml`
- 本地覆盖可以添加到`local_config.yaml`
- 特定模块的配置在配置目录中可用

配置调整示例：

```yaml
# 设置日志记录详细程度
general:
  log_level: "info"  # 选项: debug, info, warning, error

# 配置扫描频率
scheduling:
  full_scan_interval: 86400  # 秒（每天）
  quick_scan_interval: 3600  # 秒（每小时）

# 调整检测灵敏度
detection:
  sensitivity: "medium"  # 选项: low, medium, high

# 启用/禁用特定模块
modules:
  cryptominer:
    enabled: true
    sensitivity: "high"
  network:
    enabled: true
  kernel:
    enabled: true
```

## 当前开发状态

截至2025年5月，SharpEye核心模块的当前实现状态如下：

| 模块 | 状态 | 测试覆盖率 |
|--------|--------|---------------|
| 文件系统完整性 | ✅ 已完成 | 95% |
| 内核模块分析 | ✅ 已完成 | 94% |
| 库检查 | ✅ 已完成 | 95% |
| 权限提升检测 | ✅ 已完成 | 94% |
| 日志分析引擎 | ✅ 已完成 | 93% |
| 加密货币挖矿检测 | ✅ 已完成 | 95% |
| 系统资源 | ✅ 已完成 | 100% |
| 用户账户 | ✅ 已完成 | 100% |
| 进程 | ✅ 已完成 | 100% |
| 网络 | ✅ 已完成 | 95% |
| 计划任务 | ✅ 已完成 | 95% |
| SSH | ✅ 已完成 | 100% |
| Rootkit检测 | ✅ 已完成 | 100% |

该项目现在拥有一个强大的CI/CD管道，使用GitHub Actions确保所有模块的代码质量和测试覆盖率。截至最新更新（2025年5月8日），所有13个模块已全部实现并经过全面测试，提供中英双语的详细文档。CI/CD系统包括自动化测试，具有对SQLite线程问题的专门处理、全面的环境验证和详细的诊断工具，确保在不同环境中的一致质量。

## 未来发展路线图

SharpEye是一个不断发展的项目，有着雄心勃勃的未来发展计划：

### 近期目标（6-12个月）

1. **扩展OS支持**：扩大兼容性，包括更多Linux发行版
2. **增强UI**：开发全面的Web界面用于可视化和管理
3. **API增强**：扩展API，实现与SIEM和安全编排工具更好的集成
4. **容器安全**：为容器环境（Docker、Kubernetes）添加专业检测
5. **云原生集成**：为主要云平台开发插件，实现无缝集成

### 中期目标（12-24个月）

1. **高级AI模型**：实现更复杂的机器学习算法进行行为分析
2. **威胁狩猎剧本**：为常见威胁狩猎场景创建自动化工作流
3. **分布式部署**：增强监控大规模环境的能力
4. **实时关联引擎**：开发实时系统，关联多个主机的事件
5. **自动响应**：添加自动威胁缓解和响应能力

### 长期愿景（2年以上）

1. **预测性安全**：超越检测，预测潜在安全问题
2. **跨平台支持**：将核心功能扩展到其他操作系统
3. **边缘计算安全**：为IoT和边缘计算环境开发专业模块
4. **行业特定模块**：为特定行业开发量身定制的安全模块
5. **安全即代码集成**：与基础设施即代码工作流无缝集成

## 如何贡献

SharpEye在社区贡献中茁壮成长。以下是您可以参与的方式：

### 贡献领域

- **代码贡献**：增强现有模块或开发新的检测能力
- **文档**：改进指南、示例和技术文档
- **测试**：帮助在不同环境和场景中测试SharpEye
- **错误报告和功能请求**：报告问题并提出改进建议
- **威胁情报**：为签名数据库和检测规则做出贡献
- **翻译**：帮助使SharpEye在更多语言中可访问

### 开始贡献

1. **Fork仓库**：首先fork SharpEye仓库
2. **设置开发环境**：按照仓库中的开发设置指南进行操作
3. **选择一个问题**：检查问题跟踪器，寻找适合初次贡献的问题
4. **进行更改**：实施您的增强或修复
5. **提交Pull Request**：将您的更改贡献回主项目
6. **加入社区**：参与讨论并帮助他人

### 代码贡献指南

- 遵循项目的编码风格和约定
- 为新功能包含测试
- 尽可能确保向后兼容性
- 全面记录您的更改
- 保持pull requests专注于单一问题或功能

### 认可计划

我们重视所有贡献，并实施了认可计划：

- **贡献者列表**：所有贡献者在项目中得到认可
- **维护者身份**：定期贡献者可能被邀请成为维护者
- **功能归属**：主要贡献在发布说明中得到认可
- **社区聚焦**：定期突出显示杰出贡献

## 社区和支持

加入我们不断壮大的社区，获取帮助、分享想法和协作：

- **GitHub讨论**：用于问题、想法和一般讨论
- **问题跟踪器**：用于错误报告和功能请求
- **文档**：中英双语全面指南和参考材料
- **Slack频道**：用于实时协作和支持
- **月度网络研讨会**：涵盖新功能、使用案例和最佳实践

### 文档资源

SharpEye提供全面的双语文档以支持全球用户：

英文文档:
- [User Guide](https://github.com/sgInnora/sharpeye/blob/main/docs/user_guide.md)
- [Module Reference](https://github.com/sgInnora/sharpeye/blob/main/docs/module_reference.md)
- [Machine Learning Analysis](https://github.com/sgInnora/sharpeye/blob/main/docs/machine_learning_analysis.md)
- [Testing Guide](https://github.com/sgInnora/sharpeye/blob/main/docs/testing.md)
- [Project Status](https://github.com/sgInnora/sharpeye/blob/main/docs/PROJECT_STATUS.md)
- [CI/CD Implementation Guide](https://github.com/sgInnora/sharpeye/blob/main/docs/CI_CD_STATUS.md)
- [Processes Module Documentation](https://github.com/sgInnora/sharpeye/blob/main/docs/modules/PROCESSES.md)
- [Rootkit Detector Documentation](https://github.com/sgInnora/sharpeye/blob/main/docs/modules/ROOTKIT_DETECTOR.md)

中文文档:
- [用户指南](https://github.com/sgInnora/sharpeye/blob/main/docs/user_guide_zh.md)
- [模块参考](https://github.com/sgInnora/sharpeye/blob/main/docs/module_reference_zh.md)
- [机器学习分析](https://github.com/sgInnora/sharpeye/blob/main/docs/machine_learning_analysis_zh.md)
- [测试指南](https://github.com/sgInnora/sharpeye/blob/main/docs/testing_zh.md)
- [项目状态](https://github.com/sgInnora/sharpeye/blob/main/docs/PROJECT_STATUS_ZH.md)
- [CI/CD实现指南](https://github.com/sgInnora/sharpeye/blob/main/docs/CI_CD_STATUS_ZH.md)
- [进程模块文档](https://github.com/sgInnora/sharpeye/blob/main/docs/modules/PROCESSES_ZH.md)
- [Rootkit检测器文档](https://github.com/sgInnora/sharpeye/blob/main/docs/modules/ROOTKIT_DETECTOR_ZH.md)

## 关于innora.ai

innora.ai专注于为现代计算环境开发高级安全解决方案。我们的团队结合了恶意软件分析、威胁情报和机器学习领域的专业知识，创造前沿安全工具，帮助组织保护其关键基础设施。

通过开源SharpEye，我们重申了通过协作、创新和知识共享为数字世界创造更安全环境的承诺。

---

## 许可证

SharpEye根据MIT许可证发布，允许广泛使用、修改和分发，同时保持对原作者的归属。

## 致谢

- innora.ai研究团队为开发这款工具所做的贡献
- 所有帮助改进该项目的贡献者和安全研究人员
- 启发SharpEye各方面的开源安全工具
- Linux社区为此工具的运行创造了基础

---

加入我们，为全球Linux系统构建更安全的未来。一起，我们可以走在不断演变的威胁之前，保护我们的数字基础设施。

今天就探索SharpEye：[https://github.com/sgInnora/sharpeye](https://github.com/sgInnora/sharpeye)
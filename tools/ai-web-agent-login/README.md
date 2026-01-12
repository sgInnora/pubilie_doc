# NAS Docker 浏览器自动化方案

> **版本**: v2.0 (Multi-CLI 深度研究优化版)
> **目标**: 在 Synology NAS 上实现微信公众号自动发布
> **研究方法**: Gemini 3.0 Pro + Codex GPT-5 + Claude Opus 4.5 协作

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    Synology DS923+ NAS                       │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │  Nginx Proxy    │    │  Firefox Agent  │                 │
│  │  (HTTPS/Auth)   │───►│  (Selenium 4.x) │                 │
│  │  Port: 8443     │    │  VNC: 7900      │                 │
│  └─────────────────┘    └─────────────────┘                 │
│                                │                             │
│                         ┌──────┴──────┐                      │
│                         │ Persistent  │                      │
│                         │   Volume    │                      │
│                         │ (Session)   │                      │
│                         └─────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## 核心特性

| 特性 | 实现方案 | 状态 |
|------|----------|------|
| **反检测** | Xvfb有头模式 + 指纹伪装 | ✅ |
| **中文支持** | fonts-noto-cjk + UTF-8 | ✅ |
| **VNC访问** | noVNC + HTTPS + BasicAuth | ✅ |
| **Session持久化** | Docker Volume + storageState | ✅ |
| **自动保活** | 4小时定时脚本 | ✅ |
| **NAS优化** | DS923+ 无GPU软件渲染 | ✅ |

## 快速开始

### 1. 初始化配置

```bash
cd /Users/anwu/Documents/code/pubilie_doc/tools/ai-web-agent-login

# 运行初始化脚本（生成证书、密码）
./setup.sh
```

### 2. 部署到 NAS

```bash
# 一键部署
./deploy_to_nas.sh
```

### 3. 首次登录

1. 打开浏览器访问: `https://192.168.80.2:9443`
2. 输入 Basic Auth 凭据 (setup.sh 中设置)
3. 在 noVNC 中打开 Firefox
4. 访问 `https://mp.weixin.qq.com`
5. 扫码登录微信公众号
6. 登录成功后 Session 自动保存

### 4. 配置定时保活

```bash
# SSH 到 NAS
ssh feng@192.168.80.2

# 编辑 crontab
crontab -e

# 添加定时任务（每4小时执行保活）
0 */4 * * * /volume1/docker/ai-web-agent-login/keep_alive.sh
```

## 目录结构

```
ai-web-agent-login/
├── docker-compose.yml      # Docker 编排配置
├── Dockerfile              # Firefox Agent 镜像
├── browser-entrypoint.sh   # 反检测启动脚本
├── nginx/
│   ├── nginx.conf          # Nginx 反向代理配置
│   ├── .htpasswd           # Basic Auth 密码
│   └── certs/              # SSL 证书
├── data/
│   ├── profile/            # Firefox 配置持久化
│   ├── credentials/        # Session 状态存储
│   ├── downloads/          # 下载文件
│   └── logs/               # 日志文件
├── setup.sh                # 初始化配置脚本
├── deploy_to_nas.sh        # NAS 部署脚本
├── keep_alive.sh           # 保活 Shell 脚本
├── keep_alive_weixin.py    # 保活 Python 脚本
├── login_manager.py        # 登录管理器
├── weixin_publisher.py     # 微信发布器
└── credential_loader.py    # 凭据加载器
```

## 支持的平台

| 平台 | 标识 | 认证方式 | 说明 |
|------|------|----------|------|
| 微信公众号 | `weixin` | 扫码 | 需要手机微信扫码 |
| 知识星球 | `zsxq` | Token | 扫码或密码登录 |
| LinkedIn | `linkedin` | Cookie | 账号密码 |
| Twitter/X | `twitter` | Cookie | 账号密码 |
| 小红书 | `xiaohongshu` | Cookie | 账号密码/扫码 |
| 知乎 | `zhihu` | Cookie | 账号密码 |
| B站 | `bilibili` | Cookie | 账号密码/扫码 |
| 抖音 | `douyin` | Cookie | 扫码 |

## 技术方案详解

### 反检测措施

1. **Xvfb 有头模式**: Headless 模式 = 秒封
2. **指纹伪装**: `navigator.webdriver=false`, 自定义 UserAgent
3. **Canvas/WebGL 噪音**: 随机噪音注入避免指纹追踪
4. **人类化操作**: Bézier 曲线鼠标轨迹

### Session 持久化策略

```
优先级:
1. Playwright storageState (Cookie + LocalStorage)
2. Docker Volume 挂载 (browser profile)
3. 定时保活防止失效
```

### 安全配置

- ✅ HTTPS (TLS 1.2/1.3)
- ✅ HTTP Basic Auth
- ✅ 端口仅绑定 127.0.0.1
- ✅ VNC 密码保护

## 凭据管理

### 本地登录

```bash
# 登录微信公众号
python login_manager.py --platform weixin --mode local

# 列出所有已保存的凭据
python login_manager.py --list
```

### 上传凭据到 NAS

```bash
# 上传微信凭据
python login_manager.py --platform weixin --mode upload

# 重新登录并上传
python login_manager.py -p weixin -m local && python login_manager.py -p weixin -m upload
```

## 故障排查

### 浏览器崩溃

```bash
# 增加共享内存 (docker-compose.yml)
shm_size: "2gb"
```

### 登录态失效

```bash
# 检查保活日志
cat /volume1/docker/ai-web-agent-login/data/logs/keep_alive.log

# 手动执行保活
./keep_alive.sh
```

### VNC 连接超时

```bash
# 检查 Nginx WebSocket 配置
# nginx.conf: proxy_read_timeout 3600s
```

## 研究来源

本方案基于 Multi-CLI 协作研究（100+ 技术来源）：

| 研究维度 | 执行者 | 关键发现 |
|----------|--------|----------|
| Docker镜像对比 | Gemini | selenium/standalone-firefox 最稳定 |
| 反检测技术 | Gemini | Xvfb + 指纹噪音注入 |
| 微信自动化 | Gemini | DrissionPage 原生反检测 |
| Session持久化 | Gemini | storageState > Cookies |
| n8n集成 | Gemini | HTTP Request + 独立服务 |
| VNC安全 | Gemini | Nginx HTTPS 反向代理 |
| NAS优化 | Gemini | MOZ_WEBRENDER=0 软件渲染 |
| Docker配置 | Codex | 完整 Compose + Dockerfile |

详细报告: `~/analysis_results/nas-browser-automation/COMPREHENSIVE_ANALYSIS_REPORT.md`

## n8n 工作流集成

### 已部署工作流

| 工作流 | 描述 | 触发方式 | 状态 |
|--------|------|----------|------|
| **X Twitter Auto Publisher** | 推文发布（含AI检测/人性化） | Webhook POST | ✅ |
| **X Twitter Auto Reply** | 自动回复提及 | 每15分钟 | ✅ |
| **X Twitter Auto Follow** | 智能关注策略 | 每6小时 | ✅ |
| **知识星球 Auto Publisher** | 知识星球发布 | Webhook POST | ✅ |
| **Substack Newsletter** | Newsletter发布 | Webhook POST | ✅ |
| **Xiaohongshu Publisher** | 小红书发布 | Webhook POST | ✅ |
| **Xiaohongshu Auto Reply** | 小红书自动回复 | 每15分钟 | ✅ |
| **AI Content Humanizer** | 内容人性化处理 | Webhook POST | ✅ |

### 工作流修复记录 (2026-01-12)

**X Twitter Auto Publisher 修复**：

| 问题 | 原因 | 修复方案 |
|------|------|----------|
| Merge Results 节点阻塞 | `combineAll` 模式等待所有输入 | 移除节点，直连 Prepare API Call |
| `crypto` 模块禁用 | n8n code节点安全限制 | 简化代码，OAuth1由HTTP节点处理 |
| HTTP GET 而非 POST | 缺少 `method` 参数 | 添加 `"method": "POST"` |

**修复后测试**：
```bash
# 测试发推（通过 n8n Webhook）
curl -X POST 'http://192.168.80.2:5678/webhook/x-publish' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Test tweet from automation"}'
```

### 工作流配置说明

> ⚠️ **n8n-workflows/ 目录中的 JSON 文件被 .gitignore 忽略**
>
> 原因：文件包含 API 密钥和敏感凭据。工作流应通过 n8n UI 或 API 管理，不通过 Git 版本控制。

**OAuth1 凭据配置**（n8n Credentials）：
- Name: `X Twitter OAuth1`
- Consumer Key/Secret: 从 Twitter Developer Portal 获取
- Access Token/Secret: 用户授权后获取

## 安全注意事项

⚠️ **凭据文件包含敏感信息**

- 不要将 `~/.ai-web-agent/credentials/` 提交到 Git
- 确保 NAS 的凭据目录权限正确（仅 feng 用户可读写）
- 定期更新凭据，不要长期使用过期的 Cookie
- n8n 工作流 JSON 文件包含 API 密钥，已添加到 .gitignore

---

**最后更新**: 2026-01-12 22:30:00 +0800
**维护者**: Claude Opus 4.5 (Multi-CLI Architect)

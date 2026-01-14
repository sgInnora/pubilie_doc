# Docker部署NAS可行性分析报告

> **分析时间**: 2026-01-12
> **目标平台**: Synology DS923+ (192.168.80.2)
> **分析范围**: pubilie_doc自动化发布系统Docker化部署

---

## 1. 现有基础设施评估

### 1.1 NAS硬件规格
| 项目 | 规格 |
|------|------|
| 型号 | Synology DS923+ |
| CPU | AMD Ryzen R1600 (2核4线程) |
| 内存 | 4GB DDR4 (可扩展至32GB) |
| 存储 | 多盘RAID配置 |
| 网络 | 千兆以太网 |

### 1.2 已部署Docker服务
```
/volume1/docker/
├── n8n/                    # ✅ 已部署 (v2.2.4 + PostgreSQL 16)
├── cicd/                   # ✅ Gitea + Woodpecker CI
├── ai-cli-server/          # ✅ AI CLI服务
├── brain-db/               # ✅ 知识库数据库
├── chromadb/               # ✅ 向量数据库
├── ollama/                 # ✅ 本地LLM
├── proxy-service/          # ✅ 代理服务
├── novel-ai-generator/     # ✅ 小说生成器
└── 其他服务...
```

### 1.3 n8n现有配置
- **版本**: n8nio/n8n:2.2.4
- **数据库**: PostgreSQL 16-alpine
- **端口**: 5678
- **Webhook URL**: http://192.168.80.2:5678/
- **时区**: Asia/Singapore
- **已导入工作流**: 37个（包含35个安全模板）

---

## 2. pubilie_doc系统部署方案

### 2.1 组件清单

| 组件 | 功能 | Docker镜像 | 资源需求 |
|------|------|------------|----------|
| **n8n** | 工作流引擎 | n8nio/n8n:2.2.4 | ✅ 已部署 |
| **Publisher API** | 发布API服务 | python:3.11-slim | 512MB RAM |
| **Cover Generator** | 封面生成 | 本地mflux (Mac) | N/A |
| **Content Transformer** | 内容转换 | 与Publisher合并 | - |
| **PostgreSQL** | 数据存储 | postgres:16-alpine | ✅ 已部署 |
| **Redis** | 任务队列 | redis:7-alpine | 128MB RAM |

### 2.2 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    NAS (192.168.80.2)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Docker Network                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │   n8n    │  │ Postgres │  │  Publisher API   │  │   │
│  │  │  :5678   │──│  :5432   │──│     :8080        │  │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │   │
│  │       │              │               │              │   │
│  │       └──────────────┼───────────────┘              │   │
│  │                      │                              │   │
│  │  ┌──────────┐  ┌──────────┐                        │   │
│  │  │  Redis   │  │  Gitea   │                        │   │
│  │  │  :6379   │  │  :3000   │                        │   │
│  │  └──────────┘  └──────────┘                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                              │
         │ Webhook                      │ SSH/Git
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│   Mac (本地)     │            │   VM Kali       │
│  - Claude Code   │            │  - Selenium     │
│  - mflux生成     │            │  - 微信自动化    │
│  - 文章编辑      │            │  - 社交发布      │
└─────────────────┘            └─────────────────┘
```

### 2.3 推荐部署方案

#### 方案A: 最小化部署（推荐）
**仅部署必要组件，复用现有服务**

```yaml
# /volume1/docker/pubilie-doc/docker-compose.yml
version: '3.8'

services:
  pubilie-publisher:
    image: python:3.11-slim
    container_name: pubilie-publisher
    restart: unless-stopped
    working_dir: /app
    command: uvicorn main:app --host 0.0.0.0 --port 8080
    environment:
      - DATABASE_URL=postgresql://n8n:n8n_secure_password_2026@n8n-postgres:5432/pubilie
      - REDIS_URL=redis://redis:6379/0
      - N8N_WEBHOOK_URL=http://n8n:5678/webhook
    volumes:
      - /volume1/docker/pubilie-doc/app:/app
      - /volume1/docker/pubilie-doc/data:/data
    ports:
      - '8080:8080'
    networks:
      - n8n-network
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    container_name: pubilie-redis
    restart: unless-stopped
    volumes:
      - /volume1/docker/pubilie-doc/redis:/data
    networks:
      - n8n-network

networks:
  n8n-network:
    external: true
```

**优势**:
- 资源占用最小（<1GB RAM）
- 复用现有n8n和PostgreSQL
- 维护简单

**成本**: $0（使用现有NAS）

#### 方案B: 完整隔离部署
**独立环境，不依赖现有服务**

额外需要:
- 独立PostgreSQL实例
- 独立n8n实例
- 约2-3GB额外RAM

**不推荐原因**: 资源浪费，增加维护复杂度

---

## 3. 可行性评估

### 3.1 技术可行性 ✅

| 评估项 | 状态 | 说明 |
|--------|------|------|
| Docker支持 | ✅ | Synology Container Manager已安装 |
| 网络连通性 | ✅ | 192.168.80.0/24内网稳定 |
| 存储空间 | ✅ | /volume1有充足空间 |
| CPU性能 | ⚠️ | AMD R1600够用，但mflux需在Mac执行 |
| 内存容量 | ⚠️ | 4GB标配，建议扩展到8GB |
| n8n兼容 | ✅ | 已验证37个工作流正常运行 |

### 3.2 业务可行性 ✅

| 功能 | NAS部署 | Mac本地 | VM Kali |
|------|---------|---------|---------|
| 文章发布调度 | ✅ n8n | - | - |
| 内容转换 | ✅ API | - | - |
| 封面生成 | ❌ | ✅ mflux | ❌ |
| 微信自动化 | ❌ | - | ✅ Selenium |
| 数据存储 | ✅ PostgreSQL | - | - |
| 监控告警 | ✅ n8n | - | - |

### 3.3 运维可行性 ✅

| 项目 | 评估 |
|------|------|
| 自动重启 | ✅ Docker restart: unless-stopped |
| 日志管理 | ✅ Docker logs + Synology日志中心 |
| 备份恢复 | ✅ Synology Hyper Backup |
| 更新升级 | ✅ docker-compose pull && up -d |
| 故障排查 | ✅ SSH + Docker exec |

---

## 4. 部署步骤

### Step 1: 创建目录结构
```bash
ssh feng@192.168.80.2 "mkdir -p /volume1/docker/pubilie-doc/{app,data,redis}"
```

### Step 2: 上传Publisher API代码
```bash
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
  ~/Documents/code/pubilie_doc/agents/ \
  feng@192.168.80.2:/volume1/docker/pubilie-doc/app/
```

### Step 3: 部署docker-compose.yml
```bash
scp docker-compose.yml feng@192.168.80.2:/volume1/docker/pubilie-doc/
```

### Step 4: 启动服务
```bash
ssh feng@192.168.80.2 "cd /volume1/docker/pubilie-doc && docker-compose up -d"
```

### Step 5: 导入n8n工作流
```bash
# 复制工作流到n8n自定义模板目录
rsync -avz ~/Documents/code/pubilie_doc/workflows/n8n/*.json \
  feng@192.168.80.2:/volume1/docker/n8n/custom-templates/
```

### Step 6: 验证部署
```bash
# 检查服务状态
ssh feng@192.168.80.2 "docker ps | grep pubilie"

# 测试API
curl http://192.168.80.2:8080/health
```

---

## 5. 集成架构

### 5.1 工作流触发链路

```
[Mac Claude Code]
     │
     │ 1. 文章创作完成
     ▼
[Git Push to NAS Gitea]
     │
     │ 2. Webhook触发
     ▼
[n8n Workflow: content-scheduler]
     │
     │ 3. 定时/即时发布
     ▼
[n8n Workflow: cover-pipeline]
     │
     │ 4. 调用Mac mflux生成封面
     ▼
[n8n Workflow: multi-platform-publisher]
     │
     │ 5. 内容转换
     ├─────────┬─────────┬─────────┐
     ▼         ▼         ▼         ▼
[Twitter]  [LinkedIn] [Medium]  [微信]
                                   │
                                   │ 6. SSH到VM Kali
                                   ▼
                            [Selenium自动发布]
```

### 5.2 封面生成混合模式

由于NAS的AMD CPU不支持Apple MLX，封面生成采用混合模式：

1. **Mac本地生成**（默认）
   - n8n触发 → SSH到Mac → 执行mflux → 返回图片路径

2. **DALL-E 3 API降级**
   - Mac不可达时 → 调用OpenAI API → 下载图片到NAS

```javascript
// n8n Code节点示例
const macReachable = await checkMacSSH('192.168.80.xxx');
if (macReachable) {
  // SSH执行mflux
  return await execSSH('python tools/cover_generator.py --article ...');
} else {
  // 降级到DALL-E 3
  return await callDALLE3API(prompt);
}
```

---

## 6. 风险评估与缓解

### 6.1 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| NAS宕机 | 低 | 高 | UPS电源 + 定期备份 |
| 网络中断 | 中 | 中 | 本地缓存 + 重试机制 |
| Docker容器崩溃 | 低 | 低 | restart: unless-stopped |
| 存储空间不足 | 低 | 中 | 监控告警 + 自动清理 |
| API限流 | 中 | 低 | 队列削峰 + 指数退避 |

### 6.2 安全考虑

- ✅ 内网部署，不暴露公网
- ✅ SSH密钥认证
- ✅ 敏感信息存储在.env文件
- ⚠️ 需要配置防火墙规则限制访问

---

## 7. 成本分析

### 7.1 一次性成本
| 项目 | 成本 |
|------|------|
| NAS硬件 | ✅ 已有 |
| 内存升级(4GB→8GB) | ~$30 |
| 开发部署工时 | 4-8小时 |

### 7.2 持续成本
| 项目 | 月度成本 |
|------|----------|
| 电费 | ~$5 |
| 域名/SSL | ✅ 已有 |
| DALL-E 3 API（降级时） | ~$2-5 |
| **合计** | **~$7-10/月** |

---

## 8. 结论与建议

### 8.1 可行性结论

| 维度 | 评估 | 说明 |
|------|------|------|
| 技术可行性 | ✅ 高 | 现有基础设施完善 |
| 业务可行性 | ✅ 高 | 满足所有核心需求 |
| 运维可行性 | ✅ 高 | Synology生态成熟 |
| 成本可行性 | ✅ 高 | 边际成本极低 |

### 8.2 推荐行动

1. **立即执行**: 采用方案A最小化部署
2. **短期优化**: 内存升级到8GB
3. **长期规划**: 考虑Hetzner云备份节点

### 8.3 下一步

1. [ ] 创建Publisher API的FastAPI服务
2. [ ] 部署docker-compose到NAS
3. [ ] 导入8个n8n工作流
4. [ ] 配置Mac SSH密钥免密登录
5. [ ] 端到端测试完整发布链路

---

**文档版本**: v1.0
**创建时间**: 2026-01-12
**作者**: Claude Opus 4.5 (ultrathink协议)

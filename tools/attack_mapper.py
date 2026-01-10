#!/usr/bin/env python3
"""
MITRE ATT&CK 自动化映射工具

自动识别文本中的攻击技术并映射到ATT&CK框架，支持:
- TTP标注 (战术、技术、过程)
- ATT&CK Navigator Layer生成
- STIX 2.1格式导出
- IOC提取与关联

版本: 1.0
创建时间: 2026-01-10
数据源: MITRE ATT&CK STIX Data (https://github.com/mitre-attack/attack-stix-data)
"""

import re
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime


# ════════════════════════════════════════════════════════════════
# ATT&CK数据定义
# ════════════════════════════════════════════════════════════════

# 战术定义 (Enterprise ATT&CK)
TACTICS = {
    "TA0043": {"name": "Reconnaissance", "cn": "侦察"},
    "TA0042": {"name": "Resource Development", "cn": "资源开发"},
    "TA0001": {"name": "Initial Access", "cn": "初始访问"},
    "TA0002": {"name": "Execution", "cn": "执行"},
    "TA0003": {"name": "Persistence", "cn": "持久化"},
    "TA0004": {"name": "Privilege Escalation", "cn": "权限提升"},
    "TA0005": {"name": "Defense Evasion", "cn": "防御规避"},
    "TA0006": {"name": "Credential Access", "cn": "凭据访问"},
    "TA0007": {"name": "Discovery", "cn": "发现"},
    "TA0008": {"name": "Lateral Movement", "cn": "横向移动"},
    "TA0009": {"name": "Collection", "cn": "收集"},
    "TA0011": {"name": "Command and Control", "cn": "命令与控制"},
    "TA0010": {"name": "Exfiltration", "cn": "数据渗出"},
    "TA0040": {"name": "Impact", "cn": "影响"}
}

# 常见技术关键词映射 (精简版，实际应用需完整STIX数据)
TECHNIQUE_KEYWORDS = {
    # 初始访问
    "T1566": {
        "name": "Phishing",
        "cn": "钓鱼",
        "tactic": "TA0001",
        "keywords": ["phishing", "spear phishing", "钓鱼", "鱼叉攻击", "邮件欺骗", "malicious attachment"]
    },
    "T1566.001": {
        "name": "Spearphishing Attachment",
        "cn": "鱼叉钓鱼附件",
        "tactic": "TA0001",
        "keywords": ["malicious attachment", "恶意附件", "weaponized document"]
    },
    "T1566.002": {
        "name": "Spearphishing Link",
        "cn": "鱼叉钓鱼链接",
        "tactic": "TA0001",
        "keywords": ["malicious link", "恶意链接", "phishing url"]
    },
    "T1190": {
        "name": "Exploit Public-Facing Application",
        "cn": "利用面向公众的应用",
        "tactic": "TA0001",
        "keywords": ["exploit", "vulnerability", "cve-", "漏洞利用", "rce", "remote code execution"]
    },
    "T1133": {
        "name": "External Remote Services",
        "cn": "外部远程服务",
        "tactic": "TA0001",
        "keywords": ["vpn", "rdp", "ssh", "远程服务", "remote access"]
    },
    # 执行
    "T1059": {
        "name": "Command and Scripting Interpreter",
        "cn": "命令和脚本解释器",
        "tactic": "TA0002",
        "keywords": ["powershell", "cmd", "bash", "python", "脚本执行", "script execution"]
    },
    "T1059.001": {
        "name": "PowerShell",
        "cn": "PowerShell",
        "tactic": "TA0002",
        "keywords": ["powershell", "ps1", "invoke-expression", "iex"]
    },
    "T1204": {
        "name": "User Execution",
        "cn": "用户执行",
        "tactic": "TA0002",
        "keywords": ["user click", "social engineering", "用户执行", "诱导点击"]
    },
    # 持久化
    "T1547": {
        "name": "Boot or Logon Autostart Execution",
        "cn": "启动或登录自启动执行",
        "tactic": "TA0003",
        "keywords": ["autostart", "startup", "registry run key", "自启动", "开机启动"]
    },
    "T1547.001": {
        "name": "Registry Run Keys / Startup Folder",
        "cn": "注册表运行键/启动文件夹",
        "tactic": "TA0003",
        "keywords": ["registry run", "hkcu\\software\\microsoft\\windows\\currentversion\\run", "startup folder"]
    },
    "T1053": {
        "name": "Scheduled Task/Job",
        "cn": "计划任务",
        "tactic": "TA0003",
        "keywords": ["scheduled task", "cron", "at job", "计划任务", "schtasks"]
    },
    "T1543": {
        "name": "Create or Modify System Process",
        "cn": "创建或修改系统进程",
        "tactic": "TA0003",
        "keywords": ["service", "daemon", "systemd", "服务创建", "windows service"]
    },
    # 权限提升
    "T1068": {
        "name": "Exploitation for Privilege Escalation",
        "cn": "利用漏洞提权",
        "tactic": "TA0004",
        "keywords": ["privilege escalation", "提权", "local privilege", "lpe", "kernel exploit"]
    },
    "T1055": {
        "name": "Process Injection",
        "cn": "进程注入",
        "tactic": "TA0004",
        "keywords": ["process injection", "dll injection", "进程注入", "代码注入", "hollowing"]
    },
    # 防御规避
    "T1027": {
        "name": "Obfuscated Files or Information",
        "cn": "混淆文件或信息",
        "tactic": "TA0005",
        "keywords": ["obfuscation", "encoding", "混淆", "编码", "packer", "crypter"]
    },
    "T1070": {
        "name": "Indicator Removal",
        "cn": "指标清除",
        "tactic": "TA0005",
        "keywords": ["log deletion", "日志删除", "clear logs", "indicator removal", "痕迹清理"]
    },
    "T1562": {
        "name": "Impair Defenses",
        "cn": "损害防御",
        "tactic": "TA0005",
        "keywords": ["disable antivirus", "关闭杀软", "disable defender", "edr bypass"]
    },
    # 凭据访问
    "T1003": {
        "name": "OS Credential Dumping",
        "cn": "操作系统凭据转储",
        "tactic": "TA0006",
        "keywords": ["credential dump", "mimikatz", "lsass", "凭据转储", "密码窃取", "hashdump"]
    },
    "T1003.001": {
        "name": "LSASS Memory",
        "cn": "LSASS内存",
        "tactic": "TA0006",
        "keywords": ["lsass", "lsass.exe", "procdump", "mimikatz"]
    },
    "T1110": {
        "name": "Brute Force",
        "cn": "暴力破解",
        "tactic": "TA0006",
        "keywords": ["brute force", "password spray", "暴力破解", "密码喷洒", "credential stuffing"]
    },
    # 发现
    "T1082": {
        "name": "System Information Discovery",
        "cn": "系统信息发现",
        "tactic": "TA0007",
        "keywords": ["system info", "systeminfo", "系统信息", "环境探测", "reconnaissance"]
    },
    "T1083": {
        "name": "File and Directory Discovery",
        "cn": "文件和目录发现",
        "tactic": "TA0007",
        "keywords": ["file discovery", "目录遍历", "dir", "ls", "file enumeration"]
    },
    # 横向移动
    "T1021": {
        "name": "Remote Services",
        "cn": "远程服务",
        "tactic": "TA0008",
        "keywords": ["lateral movement", "横向移动", "psexec", "wmi", "winrm", "ssh lateral"]
    },
    "T1021.002": {
        "name": "SMB/Windows Admin Shares",
        "cn": "SMB/Windows管理共享",
        "tactic": "TA0008",
        "keywords": ["smb", "admin$", "c$", "ipc$", "windows share"]
    },
    # 命令与控制
    "T1071": {
        "name": "Application Layer Protocol",
        "cn": "应用层协议",
        "tactic": "TA0011",
        "keywords": ["c2", "c&c", "command and control", "命令控制", "beacon", "callback"]
    },
    "T1071.001": {
        "name": "Web Protocols",
        "cn": "Web协议",
        "tactic": "TA0011",
        "keywords": ["http c2", "https c2", "web c2", "cobaltstrike"]
    },
    "T1105": {
        "name": "Ingress Tool Transfer",
        "cn": "入口工具传输",
        "tactic": "TA0011",
        "keywords": ["download payload", "下载载荷", "certutil", "bitsadmin", "wget", "curl download"]
    },
    # 数据渗出
    "T1041": {
        "name": "Exfiltration Over C2 Channel",
        "cn": "通过C2通道渗出",
        "tactic": "TA0010",
        "keywords": ["data exfiltration", "数据外泄", "exfil", "data theft"]
    },
    # 影响
    "T1486": {
        "name": "Data Encrypted for Impact",
        "cn": "数据加密破坏",
        "tactic": "TA0040",
        "keywords": ["ransomware", "勒索软件", "encrypt files", "加密文件", "ransom"]
    },
    "T1489": {
        "name": "Service Stop",
        "cn": "服务停止",
        "tactic": "TA0040",
        "keywords": ["service stop", "停止服务", "kill process", "taskkill"]
    }
}

# APT组织映射
APT_GROUPS = {
    "APT28": {"aliases": ["Fancy Bear", "Sofacy", "STRONTIUM"], "country": "Russia"},
    "APT29": {"aliases": ["Cozy Bear", "NOBELIUM", "The Dukes"], "country": "Russia"},
    "APT1": {"aliases": ["Comment Crew", "Unit 61398"], "country": "China"},
    "APT10": {"aliases": ["Stone Panda", "MenuPass"], "country": "China"},
    "APT32": {"aliases": ["OceanLotus", "SeaLotus"], "country": "Vietnam"},
    "APT33": {"aliases": ["Elfin", "Magnallium"], "country": "Iran"},
    "APT34": {"aliases": ["OilRig", "Helix Kitten"], "country": "Iran"},
    "APT41": {"aliases": ["Winnti", "Barium"], "country": "China"},
    "Lazarus": {"aliases": ["Hidden Cobra", "ZINC", "Labyrinth Chollima"], "country": "North Korea"},
    "Kimsuky": {"aliases": ["Velvet Chollima", "Thallium"], "country": "North Korea"}
}


@dataclass
class Technique:
    """ATT&CK技术"""
    technique_id: str
    name: str
    name_cn: str
    tactic_id: str
    tactic_name: str
    confidence: float  # 0-1
    matched_keywords: List[str] = field(default_factory=list)
    context: str = ""  # 匹配上下文


@dataclass
class IOC:
    """威胁指标"""
    ioc_type: str  # ip, domain, hash, email, url
    value: str
    context: str = ""


@dataclass
class ATTACKMapping:
    """ATT&CK映射结果"""
    file_path: str
    timestamp: str

    # 识别的技术
    techniques: List[Technique] = field(default_factory=list)
    tactics_used: List[str] = field(default_factory=list)

    # APT组织关联
    apt_groups: List[str] = field(default_factory=list)

    # IOC
    iocs: List[IOC] = field(default_factory=list)

    # 统计
    total_techniques: int = 0
    coverage_score: float = 0.0  # ATT&CK覆盖度

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "techniques": [
                {
                    "id": t.technique_id,
                    "name": t.name,
                    "name_cn": t.name_cn,
                    "tactic": t.tactic_name,
                    "confidence": t.confidence,
                    "keywords": t.matched_keywords
                }
                for t in self.techniques
            ],
            "tactics_used": self.tactics_used,
            "apt_groups": self.apt_groups,
            "iocs": [{"type": i.ioc_type, "value": i.value} for i in self.iocs],
            "total_techniques": self.total_techniques,
            "coverage_score": self.coverage_score
        }


# ════════════════════════════════════════════════════════════════
# ATT&CK映射器
# ════════════════════════════════════════════════════════════════

class ATTACKMapper:
    """MITRE ATT&CK自动映射器"""

    def __init__(self, stix_path: Optional[str] = None):
        """
        初始化映射器

        Args:
            stix_path: 可选的STIX数据文件路径
        """
        self.techniques = TECHNIQUE_KEYWORDS
        self.tactics = TACTICS
        self.apt_groups = APT_GROUPS

        # 如果提供了STIX路径，尝试加载
        if stix_path:
            self._load_stix_data(stix_path)

    def _load_stix_data(self, stix_path: str):
        """加载STIX格式的ATT&CK数据"""
        path = Path(stix_path)
        if not path.exists():
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                stix_data = json.load(f)

            # 解析STIX对象
            for obj in stix_data.get("objects", []):
                if obj.get("type") == "attack-pattern":
                    # 提取技术信息
                    ext_refs = obj.get("external_references", [])
                    for ref in ext_refs:
                        if ref.get("source_name") == "mitre-attack":
                            tech_id = ref.get("external_id", "")
                            if tech_id and tech_id not in self.techniques:
                                self.techniques[tech_id] = {
                                    "name": obj.get("name", ""),
                                    "cn": obj.get("name", ""),  # 需要中文映射
                                    "tactic": "",
                                    "keywords": [obj.get("name", "").lower()]
                                }
        except Exception as e:
            print(f"Warning: Failed to load STIX data: {e}")

    def identify_techniques(self, text: str) -> List[Technique]:
        """
        从文本中识别ATT&CK技术

        Args:
            text: 要分析的文本

        Returns:
            识别到的技术列表
        """
        text_lower = text.lower()
        identified = []
        seen_ids = set()

        for tech_id, tech_info in self.techniques.items():
            matched_keywords = []
            for keyword in tech_info["keywords"]:
                if keyword.lower() in text_lower:
                    matched_keywords.append(keyword)

            if matched_keywords:
                # 避免重复（子技术和父技术）
                parent_id = tech_id.split(".")[0]
                if tech_id not in seen_ids:
                    # 计算置信度：匹配关键词越多，置信度越高
                    confidence = min(1.0, len(matched_keywords) * 0.3 + 0.2)

                    # 提取上下文
                    context = self._extract_context(text, matched_keywords[0])

                    tactic_id = tech_info.get("tactic", "")
                    tactic_name = self.tactics.get(tactic_id, {}).get("name", "Unknown")

                    technique = Technique(
                        technique_id=tech_id,
                        name=tech_info["name"],
                        name_cn=tech_info.get("cn", tech_info["name"]),
                        tactic_id=tactic_id,
                        tactic_name=tactic_name,
                        confidence=confidence,
                        matched_keywords=matched_keywords,
                        context=context[:200]
                    )
                    identified.append(technique)
                    seen_ids.add(tech_id)

        # 按置信度排序
        identified.sort(key=lambda x: x.confidence, reverse=True)
        return identified

    def _extract_context(self, text: str, keyword: str, window: int = 100) -> str:
        """提取关键词周围的上下文"""
        text_lower = text.lower()
        pos = text_lower.find(keyword.lower())
        if pos == -1:
            return ""

        start = max(0, pos - window)
        end = min(len(text), pos + len(keyword) + window)
        return text[start:end].strip()

    def identify_apt_groups(self, text: str) -> List[str]:
        """识别文本中提到的APT组织"""
        text_lower = text.lower()
        found_groups = []

        for group_name, info in self.apt_groups.items():
            # 检查组织名
            if group_name.lower() in text_lower:
                found_groups.append(group_name)
                continue

            # 检查别名
            for alias in info.get("aliases", []):
                if alias.lower() in text_lower:
                    found_groups.append(group_name)
                    break

        return list(set(found_groups))

    def extract_iocs(self, text: str) -> List[IOC]:
        """从文本中提取IOC"""
        iocs = []

        # IP地址
        ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        for match in re.finditer(ip_pattern, text):
            ip = match.group()
            # 过滤常见内网IP和特殊IP
            if not ip.startswith(('10.', '192.168.', '127.', '0.')):
                iocs.append(IOC(ioc_type="ip", value=ip))

        # 域名 (简化版)
        domain_pattern = r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+(?:com|net|org|io|ru|cn|xyz|top|info|biz)\b'
        for match in re.finditer(domain_pattern, text, re.IGNORECASE):
            domain = match.group().lower()
            # 过滤常见安全域名
            if not any(safe in domain for safe in ['google.', 'microsoft.', 'github.', 'mitre.']):
                iocs.append(IOC(ioc_type="domain", value=domain))

        # 哈希值
        # MD5
        md5_pattern = r'\b[a-fA-F0-9]{32}\b'
        for match in re.finditer(md5_pattern, text):
            iocs.append(IOC(ioc_type="md5", value=match.group().lower()))

        # SHA256
        sha256_pattern = r'\b[a-fA-F0-9]{64}\b'
        for match in re.finditer(sha256_pattern, text):
            iocs.append(IOC(ioc_type="sha256", value=match.group().lower()))

        # 去重
        seen = set()
        unique_iocs = []
        for ioc in iocs:
            key = f"{ioc.ioc_type}:{ioc.value}"
            if key not in seen:
                seen.add(key)
                unique_iocs.append(ioc)

        return unique_iocs

    def map_text(self, text: str, file_path: str = "") -> ATTACKMapping:
        """
        完整映射文本到ATT&CK框架

        Args:
            text: 要分析的文本
            file_path: 文件路径（可选）

        Returns:
            ATTACKMapping结果
        """
        # 识别技术
        techniques = self.identify_techniques(text)

        # 统计战术
        tactics_used = list(set(t.tactic_id for t in techniques if t.tactic_id))

        # 识别APT组织
        apt_groups = self.identify_apt_groups(text)

        # 提取IOC
        iocs = self.extract_iocs(text)

        # 计算覆盖度
        coverage = len(tactics_used) / len(self.tactics) if self.tactics else 0

        return ATTACKMapping(
            file_path=file_path,
            timestamp=datetime.now().isoformat(),
            techniques=techniques,
            tactics_used=tactics_used,
            apt_groups=apt_groups,
            iocs=iocs,
            total_techniques=len(techniques),
            coverage_score=coverage
        )

    def generate_navigator_layer(self, mapping: ATTACKMapping, name: str = "Auto-Generated Layer") -> Dict:
        """
        生成ATT&CK Navigator Layer

        Args:
            mapping: ATT&CK映射结果
            name: Layer名称

        Returns:
            Navigator Layer JSON dict
        """
        # Navigator Layer格式
        layer = {
            "name": name,
            "versions": {
                "attack": "14",
                "navigator": "4.9.1",
                "layer": "4.5"
            },
            "domain": "enterprise-attack",
            "description": f"Auto-generated from {mapping.file_path}",
            "filters": {
                "platforms": ["Windows", "Linux", "macOS"]
            },
            "sorting": 0,
            "layout": {
                "layout": "side",
                "aggregateFunction": "average",
                "showID": True,
                "showName": True,
                "showAggregateScores": False,
                "countUnscored": False
            },
            "hideDisabled": False,
            "techniques": [],
            "gradient": {
                "colors": ["#ff6666", "#ffff66", "#66ff66"],
                "minValue": 0,
                "maxValue": 100
            },
            "legendItems": [],
            "metadata": [],
            "links": [],
            "showTacticRowBackground": False,
            "tacticRowBackground": "#dddddd",
            "selectTechniquesAcrossTactics": True,
            "selectSubtechniquesWithParent": False
        }

        # 添加技术
        for tech in mapping.techniques:
            layer_tech = {
                "techniqueID": tech.technique_id,
                "tactic": tech.tactic_name.lower().replace(" ", "-"),
                "color": self._confidence_to_color(tech.confidence),
                "comment": f"Keywords: {', '.join(tech.matched_keywords)}",
                "enabled": True,
                "metadata": [],
                "links": [],
                "showSubtechniques": False
            }
            layer["techniques"].append(layer_tech)

        return layer

    def _confidence_to_color(self, confidence: float) -> str:
        """将置信度转换为颜色"""
        if confidence >= 0.8:
            return "#ff6666"  # 红色 - 高置信度
        elif confidence >= 0.5:
            return "#ffaa66"  # 橙色 - 中置信度
        else:
            return "#ffff66"  # 黄色 - 低置信度

    def export_stix_bundle(self, mapping: ATTACKMapping) -> Dict:
        """
        导出STIX 2.1格式的Bundle

        Args:
            mapping: ATT&CK映射结果

        Returns:
            STIX Bundle dict
        """
        import uuid

        bundle = {
            "type": "bundle",
            "id": f"bundle--{uuid.uuid4()}",
            "objects": []
        }

        # 添加Report对象
        report = {
            "type": "report",
            "spec_version": "2.1",
            "id": f"report--{uuid.uuid4()}",
            "created": mapping.timestamp,
            "modified": mapping.timestamp,
            "name": f"Analysis of {Path(mapping.file_path).name}" if mapping.file_path else "Threat Analysis",
            "published": mapping.timestamp,
            "object_refs": []
        }

        # 添加Attack Pattern对象（技术）
        for tech in mapping.techniques:
            attack_pattern = {
                "type": "attack-pattern",
                "spec_version": "2.1",
                "id": f"attack-pattern--{uuid.uuid4()}",
                "created": mapping.timestamp,
                "modified": mapping.timestamp,
                "name": tech.name,
                "description": f"MITRE ATT&CK Technique {tech.technique_id}",
                "external_references": [
                    {
                        "source_name": "mitre-attack",
                        "external_id": tech.technique_id,
                        "url": f"https://attack.mitre.org/techniques/{tech.technique_id.replace('.', '/')}/"
                    }
                ],
                "kill_chain_phases": [
                    {
                        "kill_chain_name": "mitre-attack",
                        "phase_name": tech.tactic_name.lower().replace(" ", "-")
                    }
                ]
            }
            bundle["objects"].append(attack_pattern)
            report["object_refs"].append(attack_pattern["id"])

        # 添加Indicator对象（IOC）
        for ioc in mapping.iocs:
            indicator = {
                "type": "indicator",
                "spec_version": "2.1",
                "id": f"indicator--{uuid.uuid4()}",
                "created": mapping.timestamp,
                "modified": mapping.timestamp,
                "name": f"{ioc.ioc_type.upper()}: {ioc.value}",
                "indicator_types": ["malicious-activity"],
                "pattern": self._ioc_to_stix_pattern(ioc),
                "pattern_type": "stix",
                "valid_from": mapping.timestamp
            }
            bundle["objects"].append(indicator)
            report["object_refs"].append(indicator["id"])

        # 添加Threat Actor对象（APT组织）
        for group in mapping.apt_groups:
            threat_actor = {
                "type": "threat-actor",
                "spec_version": "2.1",
                "id": f"threat-actor--{uuid.uuid4()}",
                "created": mapping.timestamp,
                "modified": mapping.timestamp,
                "name": group,
                "aliases": self.apt_groups.get(group, {}).get("aliases", []),
                "threat_actor_types": ["nation-state"]
            }
            bundle["objects"].append(threat_actor)
            report["object_refs"].append(threat_actor["id"])

        bundle["objects"].append(report)
        return bundle

    def _ioc_to_stix_pattern(self, ioc: IOC) -> str:
        """将IOC转换为STIX Pattern"""
        type_mapping = {
            "ip": "ipv4-addr:value",
            "domain": "domain-name:value",
            "md5": "file:hashes.MD5",
            "sha256": "file:hashes.'SHA-256'",
            "url": "url:value",
            "email": "email-addr:value"
        }
        stix_type = type_mapping.get(ioc.ioc_type, "artifact:payload_bin")
        return f"[{stix_type} = '{ioc.value}']"

    def format_report(self, mapping: ATTACKMapping) -> str:
        """格式化Markdown报告"""
        output = "## 🎯 MITRE ATT&CK 映射报告\n\n"

        output += f"**文件**: {mapping.file_path or 'N/A'}\n"
        output += f"**时间**: {mapping.timestamp}\n"
        output += f"**识别技术数**: {mapping.total_techniques}\n"
        output += f"**战术覆盖度**: {mapping.coverage_score:.1%}\n\n"

        # APT组织
        if mapping.apt_groups:
            output += "### 🕵️ 关联APT组织\n\n"
            for group in mapping.apt_groups:
                info = self.apt_groups.get(group, {})
                aliases = ", ".join(info.get("aliases", [])[:3])
                country = info.get("country", "Unknown")
                output += f"- **{group}** ({country}): {aliases}\n"
            output += "\n"

        # 技术列表
        if mapping.techniques:
            output += "### ⚔️ 识别的ATT&CK技术\n\n"
            output += "| ID | 技术名称 | 战术 | 置信度 | 关键词 |\n"
            output += "|----|---------:|------|--------|--------|\n"

            for tech in mapping.techniques[:20]:  # 最多显示20个
                keywords = ", ".join(tech.matched_keywords[:3])
                output += f"| [{tech.technique_id}](https://attack.mitre.org/techniques/{tech.technique_id.replace('.', '/')}) "
                output += f"| {tech.name_cn} | {tech.tactic_name} | {tech.confidence:.0%} | {keywords} |\n"
            output += "\n"

        # 战术分布
        if mapping.tactics_used:
            output += "### 📊 战术分布\n\n"
            for tactic_id in mapping.tactics_used:
                tactic_info = self.tactics.get(tactic_id, {})
                tech_count = sum(1 for t in mapping.techniques if t.tactic_id == tactic_id)
                output += f"- **{tactic_info.get('cn', tactic_id)}** ({tactic_info.get('name', '')}): {tech_count}个技术\n"
            output += "\n"

        # IOC
        if mapping.iocs:
            output += "### 🔍 提取的IOC\n\n"
            output += "| 类型 | 值 |\n"
            output += "|------|----|\n"
            for ioc in mapping.iocs[:10]:  # 最多显示10个
                output += f"| {ioc.ioc_type.upper()} | `{ioc.value}` |\n"
            if len(mapping.iocs) > 10:
                output += f"\n*（共{len(mapping.iocs)}个IOC，仅显示前10个）*\n"
            output += "\n"

        return output


def main():
    parser = argparse.ArgumentParser(description='MITRE ATT&CK自动映射工具')
    parser.add_argument('file', help='要分析的文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--format', '-f', choices=['report', 'json', 'navigator', 'stix'],
                        default='report', help='输出格式')
    parser.add_argument('--stix-data', help='ATT&CK STIX数据文件路径（可选）')
    parser.add_argument('--layer-name', default='Auto-Generated', help='Navigator Layer名称')

    args = parser.parse_args()

    # 读取文件
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"错误: 文件不存在 - {args.file}")
        return 1

    content = file_path.read_text(encoding='utf-8')

    # 初始化映射器
    mapper = ATTACKMapper(stix_path=args.stix_data)

    # 执行映射
    mapping = mapper.map_text(content, str(file_path))

    # 生成输出
    if args.format == 'report':
        output = mapper.format_report(mapping)
    elif args.format == 'json':
        output = json.dumps(mapping.to_dict(), ensure_ascii=False, indent=2)
    elif args.format == 'navigator':
        layer = mapper.generate_navigator_layer(mapping, args.layer_name)
        output = json.dumps(layer, ensure_ascii=False, indent=2)
    elif args.format == 'stix':
        bundle = mapper.export_stix_bundle(mapping)
        output = json.dumps(bundle, ensure_ascii=False, indent=2)

    # 输出
    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"输出已保存到: {args.output}")
    else:
        print(output)

    return 0


if __name__ == '__main__':
    exit(main())

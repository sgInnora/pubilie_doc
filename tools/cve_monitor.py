#!/usr/bin/env python3
"""
CVE监控工具 (E10)
=================

监控最新CVE漏洞，识别高影响力安全事件，生成文章选题建议。

数据源:
- NVD (National Vulnerability Database) API
- CISA KEV (Known Exploited Vulnerabilities)
- GitHub Security Advisories

证据来源:
- NVD API 2.0: https://nvd.nist.gov/developers/vulnerabilities
- CISA KEV: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
- GitHub Advisory DB: https://github.com/advisories

作者: Claude Opus 4.5 (Ultrathink Protocol v2.7)
创建时间: 2026-01-10
"""

import os
import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import time
import re

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVESeverity(Enum):
    """CVE严重程度"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class CVESource(Enum):
    """CVE数据源"""
    NVD = "nvd"
    CISA_KEV = "cisa_kev"
    GITHUB = "github"


@dataclass
class CVEItem:
    """CVE条目"""
    cve_id: str
    title: str
    description: str
    severity: CVESeverity
    cvss_score: float = 0.0
    source: CVESource = CVESource.NVD
    published_date: str = ""
    last_modified: str = ""
    affected_products: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    cwe_ids: List[str] = field(default_factory=list)
    is_exploited: bool = False  # CISA KEV标记
    exploit_available: bool = False
    tags: List[str] = field(default_factory=list)
    article_priority: str = "low"  # low/medium/high/critical

    def to_dict(self) -> dict:
        """转换为字典"""
        result = asdict(self)
        result["severity"] = self.severity.value
        result["source"] = self.source.value
        return result


@dataclass
class CVEReport:
    """CVE监控报告"""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    period_start: str = ""
    period_end: str = ""
    total_cves: int = 0
    critical_count: int = 0
    high_count: int = 0
    exploited_count: int = 0
    cves: List[CVEItem] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "generated_at": self.generated_at,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "summary": {
                "total": self.total_cves,
                "critical": self.critical_count,
                "high": self.high_count,
                "exploited": self.exploited_count
            },
            "cves": [cve.to_dict() for cve in self.cves],
            "recommendations": self.recommendations
        }


class CVEMonitor:
    """
    CVE监控器

    监控最新CVE漏洞，识别高价值写作选题。

    使用示例:
    ```python
    monitor = CVEMonitor(nvd_api_key="your-api-key")

    # 获取最近7天的高危CVE
    report = monitor.scan_recent(days=7, min_severity=CVESeverity.HIGH)

    # 获取写作推荐
    recommendations = monitor.get_article_recommendations(limit=5)
    ```
    """

    # 高关注产品列表（文章价值高）
    HIGH_INTEREST_PRODUCTS = [
        "microsoft", "windows", "office", "exchange", "azure",
        "apple", "ios", "macos", "safari",
        "google", "chrome", "android", "kubernetes",
        "linux", "kernel", "ubuntu", "redhat",
        "cisco", "palo alto", "fortinet", "juniper",
        "vmware", "esxi", "vcenter",
        "oracle", "java", "weblogic",
        "apache", "tomcat", "struts", "log4j",
        "openssl", "ssh", "nginx",
        "wordpress", "drupal", "joomla",
        "aws", "s3", "ec2", "lambda"
    ]

    # 高关注CWE类型
    HIGH_INTEREST_CWE = {
        "CWE-78": "OS命令注入",
        "CWE-79": "跨站脚本(XSS)",
        "CWE-89": "SQL注入",
        "CWE-94": "代码注入",
        "CWE-119": "缓冲区溢出",
        "CWE-125": "越界读取",
        "CWE-200": "信息泄露",
        "CWE-287": "认证绕过",
        "CWE-352": "CSRF",
        "CWE-416": "Use-After-Free",
        "CWE-434": "文件上传",
        "CWE-502": "反序列化",
        "CWE-611": "XXE",
        "CWE-787": "越界写入",
        "CWE-918": "SSRF"
    }

    def __init__(
        self,
        nvd_api_key: Optional[str] = None,
        cache_ttl: int = 3600
    ):
        self.nvd_api_key = nvd_api_key or os.getenv("NVD_API_KEY")
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._kev_list: Set[str] = set()

        if HAS_REQUESTS:
            self.session = requests.Session()
            if self.nvd_api_key:
                self.session.headers["apiKey"] = self.nvd_api_key
        else:
            self.session = None
            logger.warning("requests库未安装")

    def _is_cache_valid(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key not in self._cache_time:
            return False
        elapsed = (datetime.now() - self._cache_time[key]).total_seconds()
        return elapsed < self.cache_ttl

    def fetch_cisa_kev(self) -> Set[str]:
        """获取CISA已知被利用漏洞列表"""
        if self._kev_list:
            return self._kev_list

        if not self.session:
            return set()

        try:
            url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
            resp = self.session.get(url, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                self._kev_list = {
                    vuln["cveID"]
                    for vuln in data.get("vulnerabilities", [])
                }
                logger.info(f"CISA KEV: 加载 {len(self._kev_list)} 个已知被利用漏洞")

        except Exception as e:
            logger.error(f"CISA KEV获取失败: {e}")

        return self._kev_list

    def fetch_nvd_cves(
        self,
        start_date: datetime,
        end_date: datetime,
        keywords: List[str] = None
    ) -> List[CVEItem]:
        """
        从NVD获取CVE列表

        Args:
            start_date: 开始日期
            end_date: 结束日期
            keywords: 关键词筛选

        Returns:
            CVE列表
        """
        if not self.session:
            return []

        cache_key = f"nvd_{start_date.date()}_{end_date.date()}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        cves = []

        try:
            # NVD API 2.0
            base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

            params = {
                "pubStartDate": start_date.strftime("%Y-%m-%dT00:00:00.000"),
                "pubEndDate": end_date.strftime("%Y-%m-%dT23:59:59.999"),
                "resultsPerPage": 100
            }

            # 添加关键词筛选
            if keywords:
                params["keywordSearch"] = " ".join(keywords)

            start_index = 0
            total_results = 1  # 初始化

            while start_index < total_results:
                params["startIndex"] = start_index

                resp = self.session.get(base_url, params=params, timeout=60)

                if resp.status_code == 403:
                    logger.warning("NVD API需要API Key，使用免费限制")
                    time.sleep(6)  # 免费用户限制
                    resp = self.session.get(base_url, params=params, timeout=60)

                if resp.status_code != 200:
                    logger.error(f"NVD API错误: {resp.status_code}")
                    break

                data = resp.json()
                total_results = data.get("totalResults", 0)

                for vuln in data.get("vulnerabilities", []):
                    cve = self._parse_nvd_cve(vuln)
                    if cve:
                        cves.append(cve)

                start_index += 100
                time.sleep(0.6)  # 遵守API限制

                # 限制最大获取数量
                if len(cves) >= 500:
                    break

            self._cache[cache_key] = cves
            self._cache_time[cache_key] = datetime.now()

            logger.info(f"NVD: 获取 {len(cves)} 个CVE")

        except Exception as e:
            logger.error(f"NVD获取失败: {e}")

        return cves

    def _parse_nvd_cve(self, vuln_data: dict) -> Optional[CVEItem]:
        """解析NVD CVE数据"""
        try:
            cve = vuln_data.get("cve", {})
            cve_id = cve.get("id", "")

            if not cve_id:
                return None

            # 获取描述
            descriptions = cve.get("descriptions", [])
            description = ""
            for desc in descriptions:
                if desc.get("lang") == "en":
                    description = desc.get("value", "")
                    break

            # 获取CVSS分数
            metrics = cve.get("metrics", {})
            cvss_score = 0.0
            severity = CVESeverity.UNKNOWN

            # 优先使用CVSS 3.1
            if "cvssMetricV31" in metrics:
                cvss_data = metrics["cvssMetricV31"][0].get("cvssData", {})
                cvss_score = cvss_data.get("baseScore", 0.0)
                severity_str = cvss_data.get("baseSeverity", "UNKNOWN")
                severity = CVESeverity[severity_str] if severity_str in CVESeverity.__members__ else CVESeverity.UNKNOWN
            elif "cvssMetricV30" in metrics:
                cvss_data = metrics["cvssMetricV30"][0].get("cvssData", {})
                cvss_score = cvss_data.get("baseScore", 0.0)
                severity_str = cvss_data.get("baseSeverity", "UNKNOWN")
                severity = CVESeverity[severity_str] if severity_str in CVESeverity.__members__ else CVESeverity.UNKNOWN

            # 获取CWE
            cwe_ids = []
            weaknesses = cve.get("weaknesses", [])
            for weakness in weaknesses:
                for desc in weakness.get("description", []):
                    if desc.get("lang") == "en":
                        cwe_ids.append(desc.get("value", ""))

            # 获取受影响产品
            affected_products = []
            configurations = cve.get("configurations", [])
            for config in configurations:
                for node in config.get("nodes", []):
                    for cpe in node.get("cpeMatch", []):
                        criteria = cpe.get("criteria", "")
                        # 解析CPE提取产品名
                        parts = criteria.split(":")
                        if len(parts) >= 5:
                            vendor = parts[3]
                            product = parts[4]
                            affected_products.append(f"{vendor}/{product}")

            affected_products = list(set(affected_products))[:10]

            # 获取参考链接
            references = [
                ref.get("url", "")
                for ref in cve.get("references", [])[:5]
            ]

            # 检查是否在KEV列表中
            is_exploited = cve_id in self._kev_list

            # 生成标签
            tags = self._generate_tags(cve_id, description, affected_products, cwe_ids)

            # 计算文章优先级
            article_priority = self._calculate_priority(
                severity, cvss_score, is_exploited, affected_products, cwe_ids
            )

            return CVEItem(
                cve_id=cve_id,
                title=f"{cve_id}: {description[:100]}..." if len(description) > 100 else f"{cve_id}: {description}",
                description=description,
                severity=severity,
                cvss_score=cvss_score,
                source=CVESource.NVD,
                published_date=cve.get("published", ""),
                last_modified=cve.get("lastModified", ""),
                affected_products=affected_products,
                references=references,
                cwe_ids=cwe_ids,
                is_exploited=is_exploited,
                tags=tags,
                article_priority=article_priority
            )

        except Exception as e:
            logger.error(f"解析CVE失败: {e}")
            return None

    def _generate_tags(
        self,
        cve_id: str,
        description: str,
        products: List[str],
        cwe_ids: List[str]
    ) -> List[str]:
        """生成标签"""
        tags = []
        desc_lower = description.lower()

        # 产品标签
        for product in products[:3]:
            vendor, prod = product.split("/") if "/" in product else ("", product)
            if vendor:
                tags.append(vendor)
            tags.append(prod)

        # CWE标签
        for cwe in cwe_ids:
            if cwe in self.HIGH_INTEREST_CWE:
                tags.append(self.HIGH_INTEREST_CWE[cwe])

        # 攻击类型标签
        attack_keywords = {
            "remote code execution": "RCE",
            "rce": "RCE",
            "privilege escalation": "提权",
            "sql injection": "SQL注入",
            "cross-site scripting": "XSS",
            "buffer overflow": "缓冲区溢出",
            "authentication bypass": "认证绕过",
            "arbitrary file": "任意文件",
            "denial of service": "DoS"
        }

        for keyword, tag in attack_keywords.items():
            if keyword in desc_lower:
                tags.append(tag)

        return list(set(tags))[:8]

    def _calculate_priority(
        self,
        severity: CVESeverity,
        cvss_score: float,
        is_exploited: bool,
        products: List[str],
        cwe_ids: List[str]
    ) -> str:
        """计算文章撰写优先级"""
        score = 0

        # 严重程度
        if severity == CVESeverity.CRITICAL:
            score += 40
        elif severity == CVESeverity.HIGH:
            score += 25
        elif severity == CVESeverity.MEDIUM:
            score += 10

        # CVSS分数
        score += cvss_score * 3

        # 已被利用
        if is_exploited:
            score += 30

        # 高关注产品
        for product in products:
            for interest in self.HIGH_INTEREST_PRODUCTS:
                if interest in product.lower():
                    score += 15
                    break

        # 高关注CWE
        for cwe in cwe_ids:
            if cwe in self.HIGH_INTEREST_CWE:
                score += 10

        # 确定优先级
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"

    def scan_recent(
        self,
        days: int = 7,
        min_severity: CVESeverity = CVESeverity.MEDIUM
    ) -> CVEReport:
        """
        扫描最近的CVE

        Args:
            days: 查询天数
            min_severity: 最低严重程度

        Returns:
            CVE报告
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 先获取KEV列表
        self.fetch_cisa_kev()

        # 获取NVD数据
        all_cves = self.fetch_nvd_cves(start_date, end_date)

        # 按严重程度筛选
        severity_order = {
            CVESeverity.CRITICAL: 4,
            CVESeverity.HIGH: 3,
            CVESeverity.MEDIUM: 2,
            CVESeverity.LOW: 1,
            CVESeverity.UNKNOWN: 0
        }
        min_order = severity_order.get(min_severity, 0)

        filtered_cves = [
            cve for cve in all_cves
            if severity_order.get(cve.severity, 0) >= min_order
        ]

        # 按优先级和CVSS排序
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        filtered_cves.sort(
            key=lambda x: (priority_order.get(x.article_priority, 0), x.cvss_score),
            reverse=True
        )

        # 统计
        critical_count = sum(1 for c in filtered_cves if c.severity == CVESeverity.CRITICAL)
        high_count = sum(1 for c in filtered_cves if c.severity == CVESeverity.HIGH)
        exploited_count = sum(1 for c in filtered_cves if c.is_exploited)

        # 生成推荐
        recommendations = self._generate_recommendations(filtered_cves)

        return CVEReport(
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_cves=len(filtered_cves),
            critical_count=critical_count,
            high_count=high_count,
            exploited_count=exploited_count,
            cves=filtered_cves,
            recommendations=recommendations
        )

    def _generate_recommendations(self, cves: List[CVEItem], limit: int = 10) -> List[Dict[str, Any]]:
        """生成写作推荐"""
        recommendations = []

        # 优先推荐高优先级CVE
        for cve in cves:
            if len(recommendations) >= limit:
                break

            if cve.article_priority in ["critical", "high"]:
                rec = {
                    "cve_id": cve.cve_id,
                    "title": cve.title,
                    "priority": cve.article_priority,
                    "severity": cve.severity.value,
                    "cvss": cve.cvss_score,
                    "is_exploited": cve.is_exploited,
                    "affected_products": cve.affected_products[:3],
                    "tags": cve.tags,
                    "suggested_angles": self._suggest_angles(cve),
                    "references": cve.references[:3]
                }
                recommendations.append(rec)

        return recommendations

    def _suggest_angles(self, cve: CVEItem) -> List[str]:
        """为CVE建议写作角度"""
        angles = []

        if cve.is_exploited:
            angles.append("在野利用分析与紧急应对指南")

        if cve.severity == CVESeverity.CRITICAL:
            angles.append("漏洞技术深度剖析")

        if any("RCE" in tag for tag in cve.tags):
            angles.append("远程代码执行攻击链分析")

        angles.extend([
            "补丁分析与修复建议",
            "企业防护策略指南"
        ])

        return angles[:3]

    def get_article_recommendations(
        self,
        days: int = 7,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        获取文章写作推荐

        Args:
            days: 查询天数
            limit: 返回数量

        Returns:
            推荐列表
        """
        report = self.scan_recent(days=days, min_severity=CVESeverity.HIGH)
        return report.recommendations[:limit]


# ============================================================================
# 主函数
# ============================================================================

def main():
    """演示CVE监控"""
    print("=" * 60)
    print("CVE监控工具 (E10) - 演示")
    print("=" * 60)

    monitor = CVEMonitor()

    # 获取CISA KEV
    print("\n正在获取CISA已知被利用漏洞列表...")
    kev = monitor.fetch_cisa_kev()
    print(f"KEV列表包含 {len(kev)} 个漏洞")

    # 扫描最近7天
    print("\n正在扫描最近7天的CVE（注意：需要NVD API Key以获取完整数据）...")
    report = monitor.scan_recent(days=7, min_severity=CVESeverity.HIGH)

    print(f"\n扫描结果:")
    print(f"  - 总CVE数: {report.total_cves}")
    print(f"  - CRITICAL: {report.critical_count}")
    print(f"  - HIGH: {report.high_count}")
    print(f"  - 已被利用: {report.exploited_count}")

    # 显示推荐
    print(f"\n文章写作推荐 (Top 5):")
    for idx, rec in enumerate(report.recommendations[:5], 1):
        print(f"\n  {idx}. [{rec['priority'].upper()}] {rec['cve_id']}")
        print(f"     CVSS: {rec['cvss']}, 严重程度: {rec['severity']}")
        if rec['is_exploited']:
            print(f"     ⚠️  已被在野利用!")
        print(f"     影响产品: {', '.join(rec['affected_products'])}")
        print(f"     建议角度: {rec['suggested_angles'][0]}")

    # 保存报告
    output_path = "/tmp/cve_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {output_path}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

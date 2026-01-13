---
name: auditing-security
description: Performs comprehensive security audits on codebases and infrastructure. Identifies OWASP Top 10 vulnerabilities, misconfigurations, and security best practice violations. Generates detailed audit reports with remediation guidance. Triggers when user mentions "security audit", "vulnerability assessment", "pentest", or "security review".
---

# Security Audit Skill

## Overview
Conducts systematic security assessments following industry frameworks (OWASP, NIST, CIS) to identify vulnerabilities, misconfigurations, and security gaps.

## Audit Scope

### Application Security
- OWASP Top 10 vulnerabilities
- Authentication/authorization flaws
- Input validation issues
- Cryptographic weaknesses
- Session management

### Infrastructure Security
- Cloud misconfigurations
- Network security gaps
- Container security
- Secrets management
- Access control

### Code Security
- Hardcoded credentials
- Insecure dependencies
- Dangerous functions
- Information leakage
- Logic flaws

## Audit Process

```
Phase 1: Reconnaissance
├── Identify technology stack
├── Map attack surface
└── Review architecture

Phase 2: Vulnerability Assessment
├── SAST (Static Analysis)
├── Dependency scanning
├── Configuration review
└── Manual code review

Phase 3: Risk Analysis
├── Classify by severity (CVSS)
├── Assess exploitability
└── Determine business impact

Phase 4: Reporting
├── Executive summary
├── Technical findings
├── Remediation roadmap
└── Compliance mapping
```

## OWASP Top 10 (2021) Checklist

| # | Category | Check Items |
|---|----------|-------------|
| A01 | Broken Access Control | Role bypass, IDOR, path traversal |
| A02 | Cryptographic Failures | Weak encryption, plaintext secrets |
| A03 | Injection | SQL, NoSQL, OS, LDAP injection |
| A04 | Insecure Design | Threat modeling gaps, security requirements |
| A05 | Security Misconfiguration | Default configs, unnecessary features |
| A06 | Vulnerable Components | Outdated dependencies, known CVEs |
| A07 | Auth Failures | Weak passwords, session issues |
| A08 | Data Integrity Failures | Insecure deserialization, CI/CD |
| A09 | Logging Failures | Missing logs, sensitive data in logs |
| A10 | SSRF | Unvalidated URLs, internal access |

## Output Format

```markdown
# Security Audit Report

**Project**: {name}
**Audit Date**: {date}
**Auditor**: Claude Code Security Audit
**Classification**: {Confidential/Internal}

## Executive Summary

**Overall Risk Level**: {Critical/High/Medium/Low}

| Severity | Count |
|----------|-------|
| Critical | X |
| High | X |
| Medium | X |
| Low | X |
| Informational | X |

**Key Findings**:
1. {Critical finding summary}
2. {High finding summary}
3. {Notable observation}

## Detailed Findings

### [CRITICAL] {Finding Title}

**ID**: SEC-001
**CVSS Score**: 9.8 (Critical)
**OWASP Category**: A03:2021 - Injection
**CWE**: CWE-89 (SQL Injection)

**Description**:
{Detailed description of the vulnerability}

**Affected Component**:
- File: `{path/to/file}`
- Line: {line_number}
- Function: `{function_name}`

**Proof of Concept**:
```
{Steps to reproduce or example payload}
```

**Impact**:
{Business and technical impact}

**Remediation**:
```{language}
// Vulnerable code
{vulnerable_code}

// Secure code
{secure_code}
```

**References**:
- {OWASP link}
- {CWE link}

---

### [HIGH] {Finding Title}
{Same structure as above}

## Remediation Roadmap

### Immediate (0-7 days)
- [ ] {Critical fix 1}
- [ ] {Critical fix 2}

### Short-term (1-4 weeks)
- [ ] {High priority fix}
- [ ] {Security configuration}

### Medium-term (1-3 months)
- [ ] {Architecture improvement}
- [ ] {Process enhancement}

## Compliance Mapping

| Finding | PCI-DSS | SOC 2 | GDPR |
|---------|---------|-------|------|
| SEC-001 | 6.5.1 | CC6.1 | Art.32 |
| SEC-002 | 8.2.1 | CC6.6 | Art.25 |

## Appendix

### A. Tools Used
- {Tool 1}: {purpose}
- {Tool 2}: {purpose}

### B. Scope Limitations
- {Limitation 1}
- {Limitation 2}
```

## Detection Rules

### YARA Rule Template
```yara
rule Hardcoded_Credentials {
    meta:
        description = "Detects potential hardcoded credentials"
        severity = "high"
    strings:
        $password = /password\s*=\s*["'][^"']+["']/ nocase
        $api_key = /api[_-]?key\s*=\s*["'][^"']+["']/ nocase
        $secret = /secret\s*=\s*["'][^"']+["']/ nocase
    condition:
        any of them
}
```

### Sigma Rule Template
```yaml
title: Suspicious SQL Query Pattern
status: experimental
logsource:
    category: application
    product: webapp
detection:
    selection:
        query|contains:
            - "' OR '"
            - "1=1"
            - "UNION SELECT"
    condition: selection
level: high
```

## Constraints
- Never exploit vulnerabilities, only identify
- Redact sensitive data in reports
- Follow responsible disclosure practices
- Verify findings before reporting
- Provide actionable remediation guidance

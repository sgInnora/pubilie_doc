---
name: reviewing-code
description: Performs comprehensive code reviews focusing on security vulnerabilities, performance bottlenecks, and maintainability issues. Analyzes code for OWASP Top 10 risks, memory leaks, and code smells. Triggers when user asks for "code review", "security check", "quality analysis", or "review my code".
---

# Code Review Skill

## Overview
This skill provides systematic code review capabilities, combining security analysis, performance optimization, and maintainability assessment into a unified workflow.

## When to Use
- Pull request reviews
- Pre-commit code checks
- Security vulnerability scanning
- Technical debt assessment
- Code quality audits

## Review Process

### 1. Security Analysis
```
Priority: CRITICAL
Focus Areas:
- SQL/NoSQL injection vulnerabilities
- Cross-site scripting (XSS) risks
- Authentication/authorization flaws
- Sensitive data exposure
- Insecure deserialization
- Known vulnerable dependencies
```

### 2. Performance Review
```
Priority: HIGH
Focus Areas:
- N+1 query patterns
- Memory leak risks
- Inefficient algorithms (O(n²) or worse)
- Unnecessary re-renders (frontend)
- Missing caching opportunities
- Database index utilization
```

### 3. Maintainability Assessment
```
Priority: MEDIUM
Focus Areas:
- Code duplication (DRY violations)
- Function/method length (>50 lines)
- Cyclomatic complexity (>10)
- Naming conventions
- Documentation coverage
- Test coverage gaps
```

## Output Format

```markdown
# Code Review Report

**File**: [filename]
**Reviewer**: Claude Code
**Date**: [date]

## Executive Summary
[2-3 sentence overview of findings]

## Critical Issues (Must Fix)
| # | Line | Issue | Severity | Recommendation |
|---|------|-------|----------|----------------|
| 1 | XX | [description] | Critical | [fix] |

## Security Findings
### [Finding Title]
- **Location**: `file:line`
- **Risk**: [High/Medium/Low]
- **Description**: [what's wrong]
- **Remediation**: [how to fix]
- **Code Example**:
```[language]
// Before (vulnerable)
[code]

// After (secure)
[code]
```

## Performance Recommendations
[List with priorities]

## Code Quality Suggestions
[List with examples]

## Summary
- Critical: X issues
- High: X issues
- Medium: X issues
- Low: X issues

**Recommended Actions**: [prioritized list]
```

## Review Checklist
- [ ] All inputs validated and sanitized
- [ ] Authentication properly implemented
- [ ] Authorization checks in place
- [ ] Sensitive data encrypted
- [ ] Error handling doesn't leak info
- [ ] Logging excludes sensitive data
- [ ] Dependencies up to date
- [ ] Tests cover critical paths

## Constraints
- Focus on actionable, specific feedback
- Provide code examples for all fixes
- Prioritize security over style issues
- Be constructive, not critical
- Respect existing code style conventions

## Examples

### Example Request
"Review this authentication function for security issues"

### Example Response
```markdown
# Code Review Report

**File**: auth.py
**Focus**: Security Analysis

## Critical Issues

### 1. SQL Injection Vulnerability
- **Location**: `auth.py:42`
- **Risk**: Critical
- **Description**: User input directly concatenated into SQL query
- **Remediation**: Use parameterized queries

```python
# Before (vulnerable)
query = f"SELECT * FROM users WHERE username = '{username}'"

# After (secure)
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, (username,))
```

### 2. Weak Password Hashing
- **Location**: `auth.py:67`
- **Risk**: High
- **Description**: MD5 used for password hashing
- **Remediation**: Use bcrypt with appropriate work factor

```python
# Before (weak)
password_hash = hashlib.md5(password.encode()).hexdigest()

# After (secure)
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))
```
```

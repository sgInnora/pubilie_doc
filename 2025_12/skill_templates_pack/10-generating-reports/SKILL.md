---
name: generating-reports
description: Creates professional reports including technical reports, executive summaries, incident reports, and status updates. Formats content for different audiences and purposes. Triggers when user asks to "write report", "create summary", "document findings", or "generate executive brief".
---

# Report Generation Skill

## Overview
Produces well-structured, audience-appropriate reports ranging from technical deep-dives to executive summaries, following established documentation standards.

## Report Types

### 1. Technical Report
Detailed technical documentation for engineering audiences

### 2. Executive Summary
High-level overview for leadership/stakeholders

### 3. Incident Report
Post-mortem documentation for issues/outages

### 4. Status Report
Regular progress updates for projects/initiatives

### 5. Research Report
Findings from investigations or evaluations

## Report Generation Process

```
Phase 1: Context Gathering
├── Identify audience
├── Define purpose
├── Gather source materials
└── Determine format

Phase 2: Structuring
├── Select appropriate template
├── Create outline
├── Organize information hierarchy
└── Plan visuals/data

Phase 3: Writing
├── Draft sections
├── Add supporting evidence
├── Include recommendations
└── Create executive summary

Phase 4: Review
├── Check accuracy
├── Verify completeness
├── Ensure clarity
└── Format consistently
```

## Templates

### Technical Report Template
```markdown
# {Report Title}

**Author**: {name}
**Date**: {date}
**Version**: {version}
**Classification**: {Internal/Confidential/Public}

## Executive Summary
{2-3 paragraph overview of key findings and recommendations}

## 1. Introduction
### 1.1 Background
{Context and reason for this report}

### 1.2 Scope
{What is and isn't covered}

### 1.3 Methodology
{How the analysis/work was conducted}

## 2. Findings
### 2.1 {Finding Category 1}
{Detailed findings with supporting data}

### 2.2 {Finding Category 2}
{Detailed findings with supporting data}

## 3. Analysis
### 3.1 {Analysis Topic}
{Interpretation of findings}

### 3.2 Implications
{What the findings mean}

## 4. Recommendations
| Priority | Recommendation | Effort | Impact |
|----------|---------------|--------|--------|
| High | {rec1} | {effort} | {impact} |
| Medium | {rec2} | {effort} | {impact} |

## 5. Conclusion
{Summary and next steps}

## Appendices
### A. Data Tables
### B. Methodology Details
### C. References
```

### Executive Summary Template
```markdown
# Executive Summary: {Topic}

**Prepared for**: {Audience}
**Date**: {date}
**Prepared by**: {name/team}

---

## Situation
{1-2 sentences describing the current state or problem}

## Key Findings
- **Finding 1**: {One sentence with key metric/fact}
- **Finding 2**: {One sentence with key metric/fact}
- **Finding 3**: {One sentence with key metric/fact}

## Impact
| Area | Current | Projected | Change |
|------|---------|-----------|--------|
| {area1} | {val} | {val} | {±X%} |
| {area2} | {val} | {val} | {±X%} |

## Recommendations
1. **{Recommendation 1}**: {Brief description} [Priority: High]
   - Investment: {cost/effort}
   - Expected Return: {benefit}

2. **{Recommendation 2}**: {Brief description} [Priority: Medium]
   - Investment: {cost/effort}
   - Expected Return: {benefit}

## Decision Requested
{Clear statement of what decision or action is needed}

## Timeline
{Key dates and milestones}

---
*Full report available: {link}*
```

### Incident Report Template
```markdown
# Incident Report

**Incident ID**: {ID}
**Severity**: {Critical/High/Medium/Low}
**Status**: {Resolved/Ongoing/Under Investigation}
**Date**: {incident_date}
**Duration**: {X hours Y minutes}

## Summary
{One paragraph describing what happened}

## Timeline

| Time (UTC) | Event |
|------------|-------|
| {HH:MM} | {First detection/alert} |
| {HH:MM} | {Response initiated} |
| {HH:MM} | {Key action taken} |
| {HH:MM} | {Resolution/Mitigation} |

## Impact

| Metric | Value |
|--------|-------|
| Users Affected | {number} |
| Duration | {time} |
| Revenue Impact | ${amount} |
| SLA Breach | {Yes/No} |

## Root Cause
{Detailed explanation of what caused the incident}

## Resolution
{What was done to fix the issue}

## Contributing Factors
1. {Factor 1}
2. {Factor 2}
3. {Factor 3}

## Action Items

| ID | Action | Owner | Due Date | Status |
|----|--------|-------|----------|--------|
| 1 | {action} | {name} | {date} | Open |
| 2 | {action} | {name} | {date} | Open |

## Lessons Learned
- {Lesson 1}
- {Lesson 2}

## Prevention Measures
- {Measure 1}
- {Measure 2}

---
**Report Author**: {name}
**Review Date**: {date}
**Approved By**: {name}
```

### Research/Evaluation Report Template
```markdown
# {Research Topic} Evaluation Report

**Author**: {name}
**Date**: {date}
**Stakeholders**: {list}

## Objective
{What question are we trying to answer?}

## Methodology
{How was the research/evaluation conducted?}

## Options Evaluated

| Option | Description |
|--------|-------------|
| A | {description} |
| B | {description} |
| C | {description} |

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| {criterion1} | {X}% | {description} |
| {criterion2} | {Y}% | {description} |
| {criterion3} | {Z}% | {description} |

## Detailed Analysis

### Option A: {Name}
**Score**: {X}/100

| Criterion | Score | Notes |
|-----------|-------|-------|
| {criterion1} | {score} | {notes} |
| {criterion2} | {score} | {notes} |

**Pros**:
- {pro1}
- {pro2}

**Cons**:
- {con1}
- {con2}

### Option B: {Name}
{Same structure as Option A}

## Comparison Matrix

| Criterion | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| {criterion1} | {score} | {score} | {score} |
| {criterion2} | {score} | {score} | {score} |
| **Weighted Total** | {total} | {total} | {total} |

## Recommendation
{Clear recommendation with justification}

## Implementation Considerations
{What would be needed to implement the recommendation}

## Risks
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| {risk1} | {prob} | {impact} | {mitigation} |
```

## Writing Guidelines
- Use active voice
- Be concise and specific
- Support claims with data
- Use consistent formatting
- Include clear recommendations
- Tailor language to audience

## Constraints
- Verify all facts and figures
- Cite sources for data
- Get stakeholder review
- Follow org style guide
- Include version history
- Mark confidential content

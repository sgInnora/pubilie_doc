---
name: managing-projects
description: Assists with project management tasks including task breakdown, timeline planning, resource allocation, and progress tracking. Creates project plans, sprints, and roadmaps. Triggers when user asks to "plan project", "create roadmap", "break down tasks", or "track progress".
---

# Project Management Skill

## Overview
Provides structured project management support including work breakdown, scheduling, resource planning, and progress monitoring using agile and traditional methodologies.

## Supported Methodologies
- Agile/Scrum
- Kanban
- Waterfall
- Hybrid approaches

## Project Planning Process

```
Phase 1: Initiation
├── Define project scope
├── Identify stakeholders
├── Set objectives and KPIs
└── Establish constraints

Phase 2: Planning
├── Create Work Breakdown Structure (WBS)
├── Estimate effort and duration
├── Allocate resources
├── Define dependencies
└── Build timeline/roadmap

Phase 3: Execution Tracking
├── Monitor progress
├── Track blockers
├── Update status
└── Manage risks

Phase 4: Reporting
├── Generate status reports
├── Calculate metrics
├── Identify trends
└── Recommend adjustments
```

## Work Breakdown Structure Template

```markdown
# Project: {Project Name}

## Epic 1: {Epic Name}
### Story 1.1: {Story Name}
- [ ] Task 1.1.1: {Task description} [Est: Xh]
- [ ] Task 1.1.2: {Task description} [Est: Xh]
- [ ] Task 1.1.3: {Task description} [Est: Xh]

### Story 1.2: {Story Name}
- [ ] Task 1.2.1: {Task description} [Est: Xh]
- [ ] Task 1.2.2: {Task description} [Est: Xh]

## Epic 2: {Epic Name}
### Story 2.1: {Story Name}
- [ ] Task 2.1.1: {Task description} [Est: Xh]
```

## Sprint Planning Template

```markdown
# Sprint {N}: {Sprint Goal}

**Duration**: {start_date} - {end_date}
**Capacity**: {X} story points
**Team**: {team_members}

## Sprint Backlog

| ID | Story | Points | Assignee | Status |
|----|-------|--------|----------|--------|
| S-1 | {story} | 5 | {name} | To Do |
| S-2 | {story} | 3 | {name} | To Do |
| S-3 | {story} | 8 | {name} | To Do |

**Total Points**: {X}
**Velocity Target**: {Y}

## Sprint Goals
1. {Primary goal}
2. {Secondary goal}

## Risks/Dependencies
- [ ] {Risk 1}: {mitigation}
- [ ] {Dependency 1}: {owner}

## Definition of Done
- [ ] Code reviewed
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Deployed to staging
```

## Roadmap Template

```markdown
# Product Roadmap: {Product Name}

## Q1 {Year}
### Milestone 1: {Name} - {Target Date}
- Feature A: {description}
- Feature B: {description}
**Success Criteria**: {metrics}

### Milestone 2: {Name} - {Target Date}
- Feature C: {description}
**Success Criteria**: {metrics}

## Q2 {Year}
### Milestone 3: {Name} - {Target Date}
- Feature D: {description}
- Feature E: {description}

## Q3-Q4 {Year}
### Future Items (Tentative)
- {Item 1}
- {Item 2}

---

## Legend
🟢 On Track | 🟡 At Risk | 🔴 Blocked | ⚪ Not Started
```

## Status Report Template

```markdown
# Project Status Report

**Project**: {name}
**Report Date**: {date}
**Reporter**: {name}

## Executive Summary
{2-3 sentence overview of project health}

## Overall Status: 🟢/🟡/🔴

## Progress This Period
- ✅ {Completed item 1}
- ✅ {Completed item 2}
- 🔄 {In progress item 1}

## Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Sprint Velocity | 30 pts | 28 pts | 🟡 |
| Bug Count | <10 | 8 | 🟢 |
| Test Coverage | 80% | 82% | 🟢 |

## Blockers & Risks

| Issue | Impact | Owner | ETA |
|-------|--------|-------|-----|
| {Issue 1} | High | {name} | {date} |

## Next Period Plan
1. {Plan item 1}
2. {Plan item 2}

## Resource Needs
- {Resource need 1}
```

## Estimation Guidelines

### Story Point Scale
| Points | Complexity | Time Equivalent |
|--------|------------|-----------------|
| 1 | Trivial | < 2 hours |
| 2 | Simple | 2-4 hours |
| 3 | Moderate | 4-8 hours |
| 5 | Complex | 1-2 days |
| 8 | Very Complex | 2-3 days |
| 13 | Epic-level | 3-5 days |

### Risk Multipliers
- New technology: 1.5x
- External dependency: 1.3x
- Unclear requirements: 1.5x
- Complex integration: 1.4x

## Constraints
- Estimates are ranges, not commitments
- Include buffer for unknowns (15-20%)
- Update plans when scope changes
- Track actual vs estimated regularly
- Communicate risks early

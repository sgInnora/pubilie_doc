---
name: collaborating-teams
description: Facilitates team collaboration through code review guidelines, PR templates, communication standards, and knowledge sharing practices. Helps establish team workflows and documentation. Triggers when user asks about "team workflow", "code review process", "PR guidelines", or "collaboration standards".
---

# Team Collaboration Skill

## Overview
Establishes effective team collaboration practices including code review processes, pull request conventions, knowledge sharing, and communication standards.

## Collaboration Areas

### 1. Code Review Process
### 2. Pull Request Guidelines
### 3. Communication Standards
### 4. Knowledge Sharing
### 5. Onboarding Documentation

## Code Review Guidelines

### Review Checklist Template
```markdown
## Code Review Checklist

### Functionality
- [ ] Code accomplishes the stated goal
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] No obvious bugs

### Code Quality
- [ ] Code is readable and self-documenting
- [ ] Functions/methods are appropriately sized
- [ ] No unnecessary complexity
- [ ] Follows project coding standards

### Security
- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] No SQL/XSS vulnerabilities
- [ ] Authentication/authorization correct

### Testing
- [ ] Unit tests included
- [ ] Tests cover edge cases
- [ ] Tests are meaningful (not just coverage)
- [ ] Integration tests if needed

### Documentation
- [ ] Public APIs documented
- [ ] Complex logic explained
- [ ] README updated if needed
- [ ] Breaking changes noted
```

### Constructive Feedback Guide
```markdown
## How to Give Constructive Code Review Feedback

### DO:
- "Consider using X because..."
- "I'd suggest Y for better readability"
- "What do you think about...?"
- "Nice approach! One small suggestion..."

### DON'T:
- "This is wrong"
- "Why would you do it this way?"
- "This doesn't make sense"
- "Obviously you should..."

### Examples:

**Instead of**: "This variable name is bad"
**Say**: "Consider renaming `d` to `daysSinceLastLogin` for clarity"

**Instead of**: "This function is too long"
**Say**: "This function handles multiple responsibilities. Consider extracting the validation logic into a separate `validateInput()` method"

**Instead of**: "Why didn't you add tests?"
**Say**: "Could you add a test case for the null input scenario? It would help prevent regressions"
```

## Pull Request Templates

### Feature PR Template
```markdown
## Description
<!-- Brief description of the changes -->

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Refactoring
- [ ] Documentation update

## Related Issues
Closes #{issue_number}

## Changes Made
-
-
-

## Screenshots (if applicable)

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] All tests passing

## Checklist
- [ ] Code follows project style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Breaking changes documented

## Notes for Reviewers
<!-- Any context that would help reviewers -->
```

### Bug Fix PR Template
```markdown
## Bug Description
<!-- What was the bug? -->

## Root Cause
<!-- What caused the bug? -->

## Fix Description
<!-- How was it fixed? -->

## How to Reproduce (before fix)
1.
2.
3.

## Verification Steps
1.
2.
3.

## Related Issues
Fixes #{issue_number}

## Testing
- [ ] Reproduced bug before fix
- [ ] Verified fix resolves issue
- [ ] Added regression test
- [ ] No new issues introduced

## Rollback Plan
<!-- How to rollback if needed -->
```

## Communication Standards

### Async Communication Template
```markdown
## Team Async Communication Guidelines

### When to Use Each Channel

| Channel | Use For | Response Time |
|---------|---------|---------------|
| Slack #general | Team announcements | 24h |
| Slack #engineering | Technical discussions | 4h |
| GitHub Issues | Bug reports, feature requests | 48h |
| GitHub PRs | Code reviews | 24h |
| Email | External communication | 48h |
| Meetings | Complex discussions, decisions | Scheduled |

### Message Structure
1. **Context**: What's the situation?
2. **Ask**: What do you need?
3. **Deadline**: When do you need it?

### Example:
> **Context**: Working on the authentication refactor (PR #123)
> **Ask**: Need review on the JWT implementation approach
> **Deadline**: Before end of sprint (Friday)
```

### Decision Documentation Template
```markdown
## Decision Record: {Title}

**Date**: {date}
**Status**: {Proposed/Accepted/Deprecated}
**Decision Makers**: {names}

### Context
{What is the issue or situation?}

### Options Considered
1. **Option A**: {description}
   - Pros: {list}
   - Cons: {list}

2. **Option B**: {description}
   - Pros: {list}
   - Cons: {list}

### Decision
{What was decided and why}

### Consequences
{What are the implications of this decision?}

### Review Date
{When should this decision be revisited?}
```

## Knowledge Sharing

### Technical Documentation Template
```markdown
# {Component/Feature} Documentation

## Overview
{What is this and why does it exist?}

## Architecture
{High-level architecture diagram or description}

## Key Concepts
- **{Concept 1}**: {explanation}
- **{Concept 2}**: {explanation}

## Getting Started
```bash
# Setup commands
```

## Common Tasks

### {Task 1}
{Step-by-step instructions}

### {Task 2}
{Step-by-step instructions}

## Troubleshooting

| Problem | Solution |
|---------|----------|
| {problem1} | {solution1} |
| {problem2} | {solution2} |

## FAQ
**Q: {question}**
A: {answer}

## Related Resources
- {Link to related doc}
- {Link to external resource}
```

### Onboarding Checklist
```markdown
# New Team Member Onboarding

## Week 1: Setup & Orientation

### Day 1
- [ ] Accounts created (GitHub, Slack, Jira, etc.)
- [ ] Introduction to team members
- [ ] Overview of team mission and projects
- [ ] Buddy assigned

### Day 2-3
- [ ] Development environment setup
- [ ] Access to repositories
- [ ] First codebase walkthrough
- [ ] Read team documentation

### Day 4-5
- [ ] First small PR (typo fix, doc update)
- [ ] Shadow a code review
- [ ] Attend team standup
- [ ] 1:1 with manager

## Week 2: First Contribution
- [ ] Pick up starter ticket
- [ ] Complete first real PR
- [ ] Conduct first code review
- [ ] Attend sprint ceremonies

## Week 3-4: Independence
- [ ] Handle tickets independently
- [ ] Participate in on-call rotation (shadow)
- [ ] Contribute to team discussion
- [ ] 30-day check-in

## Resources
- Team wiki: {link}
- Tech documentation: {link}
- Team calendar: {link}
```

## Meeting Guidelines

### Effective Meeting Template
```markdown
## Meeting: {Title}

**Date**: {date}
**Attendees**: {list}
**Duration**: {time}

### Agenda
1. {Topic 1} (X min) - {owner}
2. {Topic 2} (Y min) - {owner}
3. Open discussion (Z min)

### Notes
{Meeting notes}

### Decisions Made
1. {Decision 1}
2. {Decision 2}

### Action Items
| Item | Owner | Due Date |
|------|-------|----------|
| {action} | {name} | {date} |

### Next Meeting
{Date/time for follow-up}
```

## Constraints
- Keep documentation up to date
- Respect async communication times
- Be inclusive in discussions
- Document decisions for future reference
- Regular feedback loops
- Continuous process improvement

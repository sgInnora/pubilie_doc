---
name: refactoring-code
description: Performs systematic code refactoring to improve maintainability, reduce technical debt, and enhance performance. Applies design patterns, SOLID principles, and clean code practices. Triggers when user asks to "refactor", "clean up code", "improve code quality", or "reduce technical debt".
---

# Code Refactoring Skill

## Overview
Systematically improves code structure without changing external behavior, applying industry-standard patterns and principles to enhance maintainability, readability, and performance.

## Refactoring Categories

### 1. Code Smells Detection
- Long methods (>20 lines)
- Large classes (>200 lines)
- Duplicate code blocks
- Deep nesting (>3 levels)
- Long parameter lists (>4 params)
- Feature envy
- Data clumps

### 2. Design Pattern Application
- Factory for object creation
- Strategy for algorithm variants
- Observer for event handling
- Decorator for behavior extension
- Adapter for interface compatibility

### 3. SOLID Principles
- **S**ingle Responsibility
- **O**pen/Closed
- **L**iskov Substitution
- **I**nterface Segregation
- **D**ependency Inversion

## Refactoring Process

```
Phase 1: Analysis
├── Run static analysis
├── Identify code smells
├── Measure complexity metrics
└── Prioritize refactoring targets

Phase 2: Planning
├── Define refactoring scope
├── Identify dependencies
├── Plan incremental changes
└── Ensure test coverage

Phase 3: Execution
├── Apply refactoring patterns
├── Run tests after each change
├── Commit atomic changes
└── Update documentation

Phase 4: Verification
├── Run full test suite
├── Compare metrics before/after
├── Review with team
└── Document improvements
```

## Common Refactoring Patterns

### Extract Method
```python
# Before
def process_order(order):
    # Validate order
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Invalid total")

    # Calculate discount
    discount = 0
    if order.customer.is_vip:
        discount = order.total * 0.1
    elif order.total > 100:
        discount = order.total * 0.05

    # Process payment
    final_total = order.total - discount
    payment_result = payment_gateway.charge(order.customer, final_total)
    return payment_result

# After
def process_order(order):
    validate_order(order)
    discount = calculate_discount(order)
    return process_payment(order, discount)

def validate_order(order):
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Invalid total")

def calculate_discount(order):
    if order.customer.is_vip:
        return order.total * 0.1
    elif order.total > 100:
        return order.total * 0.05
    return 0

def process_payment(order, discount):
    final_total = order.total - discount
    return payment_gateway.charge(order.customer, final_total)
```

### Replace Conditional with Polymorphism
```python
# Before
class Bird:
    def get_speed(self):
        if self.type == "european":
            return self.base_speed
        elif self.type == "african":
            return self.base_speed - self.load_factor * self.coconuts
        elif self.type == "norwegian_blue":
            return 0 if self.is_nailed else self.base_speed

# After
class Bird:
    def get_speed(self):
        raise NotImplementedError

class EuropeanBird(Bird):
    def get_speed(self):
        return self.base_speed

class AfricanBird(Bird):
    def get_speed(self):
        return self.base_speed - self.load_factor * self.coconuts

class NorwegianBlueBird(Bird):
    def get_speed(self):
        return 0 if self.is_nailed else self.base_speed
```

### Introduce Parameter Object
```typescript
// Before
function createReport(
    startDate: Date,
    endDate: Date,
    department: string,
    format: string,
    includeCharts: boolean,
    emailRecipients: string[]
): Report { /* ... */ }

// After
interface ReportConfig {
    dateRange: { start: Date; end: Date };
    department: string;
    format: 'pdf' | 'excel' | 'html';
    includeCharts: boolean;
    emailRecipients: string[];
}

function createReport(config: ReportConfig): Report { /* ... */ }
```

## Output Format

```markdown
# Refactoring Report

**File**: {filename}
**Date**: {date}
**Complexity Before**: {before_metrics}
**Complexity After**: {after_metrics}

## Changes Summary

| Change Type | Count | Impact |
|-------------|-------|--------|
| Extract Method | X | Reduced method length |
| Rename | X | Improved readability |
| Move | X | Better organization |

## Detailed Changes

### 1. {Change Title}
**Pattern Applied**: {pattern_name}
**Reason**: {why_this_refactoring}

```{language}
// Before
{original_code}

// After
{refactored_code}
```

**Benefits**:
- {benefit_1}
- {benefit_2}

## Metrics Comparison

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Cyclomatic Complexity | X | Y | -Z% |
| Lines of Code | X | Y | -Z% |
| Method Count | X | Y | +Z |
| Test Coverage | X% | Y% | +Z% |

## Recommendations
1. {Next refactoring suggestion}
2. {Technical debt item}
```

## Constraints
- Never change external behavior
- Ensure tests pass before and after
- Make atomic, incremental changes
- Preserve backward compatibility
- Document breaking changes if any
- Keep refactoring scope focused

---
name: analyzing-data
description: Performs data analysis including statistical analysis, trend identification, anomaly detection, and visualization recommendations. Works with CSV, JSON, SQL query results, and log files. Triggers when user asks to "analyze data", "find patterns", "generate statistics", or "explore dataset".
---

# Data Analysis Skill

## Overview
Provides comprehensive data analysis capabilities including descriptive statistics, trend analysis, anomaly detection, and actionable insights from structured and semi-structured data.

## Supported Data Formats
- CSV/TSV files
- JSON/JSONL
- SQL query results
- Log files (structured)
- Excel exports

## Analysis Process

```
Phase 1: Data Understanding
├── Assess data shape and structure
├── Identify data types
├── Check for missing values
├── Detect outliers
└── Understand distributions

Phase 2: Exploratory Analysis
├── Calculate descriptive statistics
├── Identify correlations
├── Detect patterns and trends
├── Segment data
└── Generate hypotheses

Phase 3: Deep Analysis
├── Statistical testing
├── Anomaly detection
├── Time series analysis
├── Comparative analysis
└── Root cause investigation

Phase 4: Insights & Reporting
├── Summarize key findings
├── Recommend visualizations
├── Suggest next steps
└── Document methodology
```

## Analysis Templates

### Descriptive Statistics Report
```markdown
# Data Analysis Report

**Dataset**: {name}
**Records**: {count}
**Analysis Date**: {date}

## Overview

| Metric | Value |
|--------|-------|
| Total Records | {n} |
| Columns | {m} |
| Time Range | {start} to {end} |
| Missing Data | {X}% |

## Numerical Columns

### {Column Name}
| Statistic | Value |
|-----------|-------|
| Mean | {value} |
| Median | {value} |
| Std Dev | {value} |
| Min | {value} |
| Max | {value} |
| 25th %ile | {value} |
| 75th %ile | {value} |

**Distribution**: {Normal/Skewed Left/Skewed Right/Bimodal}
**Outliers**: {count} detected (>{threshold})

## Categorical Columns

### {Column Name}
| Value | Count | Percentage |
|-------|-------|------------|
| {val1} | {n} | {X}% |
| {val2} | {n} | {Y}% |
| Other | {n} | {Z}% |
```

### Trend Analysis Template
```markdown
# Trend Analysis Report

## Time Series Overview
- **Period**: {start} to {end}
- **Granularity**: {daily/weekly/monthly}
- **Data Points**: {n}

## Trend Summary

| Metric | Start | End | Change | % Change |
|--------|-------|-----|--------|----------|
| {metric1} | {val} | {val} | {diff} | {pct}% |
| {metric2} | {val} | {val} | {diff} | {pct}% |

## Pattern Detection
- **Seasonality**: {Yes/No} - {pattern description}
- **Cyclical Pattern**: {description}
- **Overall Trend**: {Increasing/Decreasing/Stable}

## Anomalies Detected
| Date | Metric | Expected | Actual | Deviation |
|------|--------|----------|--------|-----------|
| {date} | {metric} | {val} | {val} | {X}σ |

## Forecast (if applicable)
Based on {method}, projected values:
- Next period: {value} (±{confidence})
- {N} periods out: {value} (±{confidence})
```

### Anomaly Detection Template
```markdown
# Anomaly Detection Report

## Methodology
- **Algorithm**: {Z-score/IQR/Isolation Forest/etc.}
- **Threshold**: {value}
- **Baseline Period**: {dates}

## Anomalies Summary
| Severity | Count | Percentage |
|----------|-------|------------|
| Critical | {n} | {X}% |
| Warning | {n} | {Y}% |
| Info | {n} | {Z}% |

## Detailed Anomalies

### Critical Anomalies
| ID | Timestamp | Metric | Value | Expected | Deviation |
|----|-----------|--------|-------|----------|-----------|
| A1 | {time} | {metric} | {val} | {exp} | {dev} |

### Root Cause Candidates
- {Anomaly ID}: {Possible cause based on correlation}
- {Anomaly ID}: {Possible cause}

## Recommendations
1. {Immediate action}
2. {Investigation needed}
3. {Monitoring adjustment}
```

## Python Analysis Snippets

### Quick Statistics
```python
import pandas as pd

def quick_stats(df):
    """Generate quick statistics for a DataFrame"""
    stats = {
        'shape': df.shape,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1e6
    }
    return stats
```

### Anomaly Detection
```python
def detect_anomalies_zscore(series, threshold=3):
    """Detect anomalies using Z-score method"""
    from scipy import stats
    z_scores = stats.zscore(series.dropna())
    anomalies = abs(z_scores) > threshold
    return series[anomalies]
```

### Trend Calculation
```python
def calculate_trend(df, date_col, value_col):
    """Calculate linear trend"""
    from scipy import stats
    df = df.sort_values(date_col)
    x = range(len(df))
    y = df[value_col].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        'slope': slope,
        'r_squared': r_value**2,
        'direction': 'increasing' if slope > 0 else 'decreasing'
    }
```

## Visualization Recommendations

| Data Type | Recommended Chart |
|-----------|------------------|
| Time series | Line chart, Area chart |
| Distribution | Histogram, Box plot |
| Comparison | Bar chart, Grouped bar |
| Correlation | Scatter plot, Heatmap |
| Composition | Pie chart, Stacked bar |
| Ranking | Horizontal bar |

## Constraints
- State assumptions clearly
- Note data quality issues
- Use appropriate statistical methods
- Avoid causation claims without evidence
- Include confidence intervals where applicable
- Document data transformations

---
name: generating-docs
description: Generates comprehensive project documentation including README, architecture docs, and user guides. Analyzes codebase structure and creates standardized documentation following best practices. Triggers when user asks to "generate docs", "create README", "document project", or "write documentation".
---

# Documentation Generator Skill

## Overview
Automatically generates professional project documentation by analyzing codebase structure, extracting key information, and applying industry-standard templates.

## Supported Documentation Types

### 1. README.md
Project overview, quick start, and essential information

### 2. ARCHITECTURE.md
System design, component relationships, data flow

### 3. API.md
Endpoint documentation, request/response schemas

### 4. CONTRIBUTING.md
Contribution guidelines, code standards, PR process

### 5. CHANGELOG.md
Version history, breaking changes, migration guides

## Generation Process

```
1. Scan Project Structure
   └── Identify key directories, entry points, config files

2. Extract Information
   └── Parse package.json/requirements.txt/go.mod
   └── Identify frameworks and dependencies
   └── Find existing documentation fragments

3. Generate Documentation
   └── Apply appropriate template
   └── Fill in extracted information
   └── Add placeholder sections for manual review

4. Quality Check
   └── Verify all links work
   └── Check code examples compile
   └── Ensure consistent formatting
```

## Templates

### README Template
```markdown
# {Project Name}

> {One-line description from package.json or first code comment}

[![License](https://img.shields.io/badge/license-{LICENSE}-blue.svg)]()
[![Version](https://img.shields.io/badge/version-{VERSION}-green.svg)]()

## Features

- {Feature 1 extracted from code/comments}
- {Feature 2}
- {Feature 3}

## Quick Start

### Prerequisites
- {Runtime} >= {version}
- {Dependency 1}
- {Dependency 2}

### Installation

```bash
# Clone the repository
git clone {repo_url}
cd {project_name}

# Install dependencies
{install_command}

# Run the application
{run_command}
```

### Basic Usage

```{language}
{usage_example}
```

## Documentation

- [Architecture](./docs/ARCHITECTURE.md)
- [API Reference](./docs/API.md)
- [Contributing](./CONTRIBUTING.md)

## Project Structure

```
{project_name}/
├── {dir1}/          # {description}
├── {dir2}/          # {description}
├── {main_file}      # {description}
└── {config_file}    # {description}
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| {VAR1} | {description} | {default} |
| {VAR2} | {description} | {default} |

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

{LICENSE_TYPE} - see [LICENSE](./LICENSE) for details.
```

### ARCHITECTURE Template
```markdown
# Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────┐
│                        {System Name}                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ {Comp1} │───▶│ {Comp2} │───▶│ {Comp3} │                 │
│  └─────────┘    └─────────┘    └─────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## Components

### {Component 1}
- **Purpose**: {description}
- **Location**: `{path}`
- **Dependencies**: {list}

### {Component 2}
- **Purpose**: {description}
- **Location**: `{path}`
- **Dependencies**: {list}

## Data Flow

1. {Step 1}
2. {Step 2}
3. {Step 3}

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | {tech} | {purpose} |
| Backend | {tech} | {purpose} |
| Database | {tech} | {purpose} |

## Design Decisions

### {Decision 1}
- **Context**: {why this decision was needed}
- **Decision**: {what was decided}
- **Consequences**: {trade-offs and implications}
```

## Output Format
Generated documentation files are placed in:
- `./README.md` - Project root
- `./docs/` - Detailed documentation
- `./CONTRIBUTING.md` - Contribution guide

## Constraints
- Never include sensitive information (API keys, passwords)
- Use relative links for internal references
- Keep examples minimal but functional
- Mark uncertain sections with `[TODO: verify]`

## Example Usage

### Request
"Generate documentation for this Python Flask project"

### Output
Creates:
- README.md with project overview
- docs/ARCHITECTURE.md with system design
- docs/API.md with endpoint documentation
- CONTRIBUTING.md with guidelines

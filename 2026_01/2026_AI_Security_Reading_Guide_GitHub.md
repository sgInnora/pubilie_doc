# 2026 AI Security Reading Guide

[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)
[![Contact](https://img.shields.io/badge/Contact-security%40innora.ai-orange)](mailto:security@innora.ai)

> A curated reading list for AI security professionals, covering adversarial attacks, machine learning security, endpoint defense, and threat intelligence methodology.

---

## 📚 Quick Reference: Core Books

| # | Title | Author(s) | Year | Difficulty | Focus Area |
|---|-------|-----------|------|------------|------------|
| 1 | **Adversarial AI Attacks, Mitigations, and Defense Strategies** | John Sotiropoulos | 2024 | ★★★★☆ | Comprehensive AI attack/defense |
| 2 | **Not with a Bug, But with a Sticker** | Siva Kumar & Anderson | 2023 | ★★★☆☆ | AI security fundamentals |
| 3 | **Evading EDR** | Matt Hand | 2023 | ★★★★★ | Endpoint security foundations |
| 4 | **Visual Threat Intelligence** | Thomas Roccia | 2023 | ★★★☆☆ | Threat intelligence methodology |

---

## 🎯 2026 AI Security Context

The AI security landscape underwent fundamental changes in 2025:

- **Adversarial attacks commercialized**: Prompt injection and adversarial patches available as services
- **LLM vulnerabilities expanded**: Jailbreaks, data poisoning, and model extraction became mainstream threats
- **Regulatory frameworks activated**: EU AI Act, NIST AI RMF, and regional regulations created compliance requirements

Traditional cybersecurity knowledge requires significant adaptation for AI security contexts. This reading list addresses the knowledge gap.

---

## 📖 Detailed Book Analysis

### 1. Adversarial AI Attacks, Mitigations, and Defense Strategies

**Author**: John Sotiropoulos (OWASP LLM Top 10 Co-Lead, US AI Safety Institute Consortium)
**Publisher**: Packt Publishing, July 2024
**Pages**: 602
**ISBN**: 9781835087985

**Coverage**:
- AI security fundamentals (predictive vs generative AI)
- Threat modeling for AI systems
- Data poisoning and adversarial samples
- Model extraction and reverse engineering
- Prompt injection attacks
- MLSecOps and secure-by-design principles

**Code Repository**: [PacktPublishing/Adversarial-AI](https://github.com/PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies)

**Technical Depth**: Complete attack-defense lifecycle with runnable code examples. Covers NIST AI RMF, ISO 42001, and EU AI Act compliance frameworks.

---

### 2. Not with a Bug, But with a Sticker

**Authors**: Ram Shankar Siva Kumar (Microsoft AI Red Team Founder), Hyrum Anderson (Former Microsoft AI Red Team Lead)
**Publisher**: Wiley, 2023
**ISBN**: 9781119883982

**Key Features**:
- Narrative-driven approach to adversarial machine learning
- Case studies from government, academia, and industry
- Accessible without advanced mathematical prerequisites
- Foreword by Bruce Schneier

**Endorsements**:
- Miles Brundage (OpenAI): "Timely overview of this emerging risk landscape"
- David Brumley (CMU Professor): "Required reading to become AI/ML literate in the field"

**Scope**: Covers the evolution from academic research to real-world attacks, including the "sticker attack" phenomenon and AI red teaming methodology.

---

### 3. Evading EDR: The Definitive Guide

**Author**: Matt Hand (SpecterOps Service Architect)
**Publisher**: No Starch Press, October 2023
**ISBN**: 9781718503342

**Relevance to AI Security**:
- AI training clusters and inference servers operate in EDR-protected environments
- Understanding EDR architecture is foundational for AI infrastructure security assessments
- Provides insight into how APTs breach systems before targeting AI assets

**Technical Topics**:
- Function-hooking DLLs and API monitoring
- Process/thread creation notifications
- Event Tracing for Windows (ETW)
- Antimalware Scan Interface (AMSI)
- Complete detection-aware attack case study

---

### 4. Visual Threat Intelligence

**Author**: Thomas Roccia (Microsoft Senior Security Researcher)
**Publisher**: Security Break, 2023
**ISBN**: 9780646879376

**Core Topics**:
- Threat intelligence lifecycle and collection frameworks
- Diamond Model of Intrusion Analysis
- Analysis of Competing Hypotheses (ACH)
- Traffic Light Protocol (TLP) for information sharing
- YARA, Sigma, and MSTICpy practical implementation
- Case studies: NotPetya, Shamoon, Sunburst

**Value for AI Security**: The methodology directly applies to tracking adversarial AI threats, APT groups targeting AI systems, and communicating complex AI attack patterns.

---

## 🗺️ Learning Paths by Background

### Path A: Traditional Security → AI Security

```
Phase 1 (Weeks 1-4):   "Not with a Bug, But with a Sticker"
                       → Foundation: AI security awareness

Phase 2 (Weeks 5-12):  "Adversarial AI Attacks, Mitigations, and Defense Strategies"
                       → Depth: Technical attack-defense skills + code labs

Phase 3 (Weeks 13-18): "Visual Threat Intelligence" + "Evading EDR"
                       → Breadth: Methodology and infrastructure security
```

### Path B: ML Engineering → Security Focus

```
Primary:    "Adversarial AI Attacks" → Leverage existing ML knowledge
Secondary:  "Not with a Bug" → Industry context and historical perspective
Optional:   "Evading EDR" → System-level security depth
```

### Path C: Red Team → AI Specialization

```
Foundation:  "Evading EDR" → Endpoint security fundamentals
Core:        "Adversarial AI Attacks" → AI-specific attack techniques
Context:     "Not with a Bug" → Industry trends and case studies
Enhancement: "Visual Threat Intelligence" → Intel and analysis capabilities
```

---

## 🔗 Companion Resources

### Standards & Frameworks

| Organization | Resource | Link |
|--------------|----------|------|
| OWASP | LLM Top 10 | [owasp.org](https://owasp.org/www-project-top-10-for-large-language-model-applications/) |
| OWASP | ML Top 10 | [owasp.org](https://owasp.org/www-project-machine-learning-security-top-10/) |
| NIST | AI Risk Management Framework | [nist.gov](https://www.nist.gov/itl/ai-risk-management-framework) |
| MITRE | ATLAS (Adversarial Threat Landscape) | [atlas.mitre.org](https://atlas.mitre.org/) |

### Communities & Conferences

- DEFCON AI Village
- Black Hat AI Summit
- ConfML (Conference on Applied Machine Learning in Information Security)
- OWASP AI Security Community

### Related Training Programs

- Practical AI Security (Practical DevSecOps)
- AI/ML Security Fundamentals (SANS)
- Adversarial Machine Learning (Coursera)

---

## 📬 Contact

**Innora Security Research Team**
Email: security@innora.ai
Website: [innora.ai](https://innora.ai)

---

## 📄 License

This reading guide is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

---

*Last updated: January 6, 2026*

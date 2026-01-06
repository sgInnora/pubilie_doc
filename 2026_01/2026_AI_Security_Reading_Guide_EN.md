# 2026 AI Security Core Books Analysis: Adversarial Attacks, Machine Learning Security, and Threat Intelligence

> **Author**: Innora Security Research Team
> **Published**: January 6, 2026
> **Contact**: security@innora.ai
> **Reading Time**: ~20 minutes

---

## Executive Summary

The AI security threat landscape underwent fundamental transformation in 2025: adversarial attack techniques transitioned from academic research to commercial applications, large language model vulnerabilities emerged at scale, and global regulatory frameworks accelerated implementation. These changes demand updated knowledge frameworks for security professionals.

This article provides an in-depth analysis of four core technical books in the AI security domain, covering **adversarial AI attacks and defense**, **machine learning system security**, **endpoint detection and AI infrastructure security**, and **threat intelligence analysis methodology**. The analysis evaluates each book across technical content, author credentials, and practical value to serve as reference for security researchers and practitioners.

### Analysis Overview

| Book | Core Domain | Technical Depth | Practical Value |
|------|-------------|-----------------|-----------------|
| Adversarial AI Attacks, Mitigations, and Defense Strategies | AI Adversarial Attacks & MLSecOps | ★★★★☆ | High |
| Not with a Bug, But with a Sticker | Machine Learning System Attacks | ★★★☆☆ | High |
| Evading EDR | Endpoint Security & AI Infrastructure | ★★★★★ | High |
| Visual Threat Intelligence | Threat Intelligence Visualization | ★★★☆☆ | Medium-High |

---

## Table of Contents

1. [2025-2026 AI Security Threat Landscape](#2025-2026-ai-security-threat-landscape)
2. [Core Books Technical Analysis](#core-books-technical-analysis)
3. [Comparative Analysis and Use Cases](#comparative-analysis-and-use-cases)
4. [Companion Resources and Standards](#companion-resources-and-standards)
5. [Conclusions](#conclusions)

---

## 2025-2026 AI Security Threat Landscape

### Key Changes in the Threat Landscape

**Commercialization of Adversarial Attacks**

In 2025, adversarial sample attacks and prompt injection techniques moved from research domains into actual attack toolchains. Attackers began packaging adversarial patch generation and prompt injection payloads as commercial services, lowering attack barriers.

**Large-Scale LLM Vulnerability Emergence**

From prompt injection to jailbreaking, from data poisoning to model extraction—LLM security issues emerged at scale in enterprise deployments. Multiple high-impact security incidents demonstrated that AI assistants could become new entry points for attackers into enterprise networks.

**Accelerated Regulatory Framework Implementation**

The EU AI Act came into effect, the NIST AI Risk Management Framework (AI RMF) gained widespread adoption, and China's interim measures for generative AI services continued to mature. Compliance requirements are driving enterprises to build systematic AI security capabilities.

### Structural Differences in Knowledge Systems

Traditional cybersecurity and AI security exhibit fundamental differences:

| Dimension | Traditional Cybersecurity | AI Security |
|-----------|---------------------------|-------------|
| Attack Surface | Input validation, authentication, network perimeter | Model weights, training data, inference APIs |
| Defense Mechanisms | WAF, IDS/IPS, access control | Adversarial training, input sanitization, model monitoring |
| Risk Assessment | CVSS scoring system | No unified standard (MITRE ATLAS exploring) |

These differences determine that AI security requires specialized knowledge systems.

---

## Core Books Technical Analysis

### 1. Adversarial AI Attacks, Mitigations, and Defense Strategies

**Basic Information**

| Attribute | Details |
|-----------|---------|
| Author | John Sotiropoulos |
| Publisher | Packt Publishing |
| Publication Date | July 2024 |
| Pages | 602 |
| ISBN | 9781835087985 |

**Author Background**

John Sotiropoulos serves as Senior Security Architect at Kainos, responsible for AI security work, with practical security experience in government, regulatory, and healthcare systems. More importantly, he holds the following key roles:

- Co-lead of OWASP LLM Top 10 project
- Core member of AI Exchange, responsible for standards alignment with other standards organizations and national cybersecurity agencies
- OWASP representative at the US AI Safety Institute Consortium

These roles provide him with authoritative understanding of AI security standards and best practices.

**Technical Content Architecture**

The book adopts an "offense-defense integration" structure, divided into three parts:

**Part One: AI Security Fundamentals (~150 pages)**
- Security differences between predictive AI and generative AI
- Unique attack surface identification methods for AI systems
- Threat modeling methodology for AI systems (extended STRIDE)
- Relationship between AI security and traditional cybersecurity

**Part Two: Adversarial Attack Techniques (~250 pages)**
- Data poisoning attacks: Backdoor injection during training
- Adversarial sample attacks: Perturbation and patch attacks on image classifiers
- Model extraction attacks: Reconstructing models through API queries
- Model reverse engineering: Inferring training data from model outputs
- Prompt injection attacks: Attack techniques targeting LLMs
- Jailbreak attacks: Bypassing LLM safety alignment

**Part Three: Defense and MLSecOps (~200 pages)**
- Secure design principles and architectural patterns
- Model validation, testing, and monitoring methods
- Integrating security into MLOps workflows (MLSecOps)
- Incident response and recovery strategies

**Technical Highlights**

1. **Threat Modeling Framework**: Provides threat modeling methodology specifically designed for AI systems—a rare systematized approach in the current market
2. **Code Implementation**: Companion GitHub repository (PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies) contains attack demonstrations and defense code
3. **Standards Alignment**: Content aligns closely with OWASP LLM Top 10 and NIST AI RMF

**Limitations**

- Some content focuses on conceptual introduction; code implementation depth for advanced attack techniques is limited
- Generative AI security content is based on mid-2024 technical state; rapid LLM evolution may require updates

---

### 2. Not with a Bug, But with a Sticker

**Basic Information**

| Attribute | Details |
|-----------|---------|
| Authors | Ram Shankar Siva Kumar, Hyrum Anderson |
| Publisher | Wiley |
| Publication Date | 2023 |
| ISBN | 9781119883982 |
| Foreword | Bruce Schneier |

**Author Background**

Both authors come from core practice positions in the AI security field:

**Ram Shankar Siva Kumar**
- Microsoft "Data Cowboy"
- Founder of Microsoft AI Red Team
- Focused on the intersection of machine learning and security

**Hyrum Anderson**
- Distinguished Engineer at Robust Intelligence
- Former leader of Microsoft AI Red Team
- Former Chief Scientist at Endgame
- Co-founder of Conference on Applied Machine Learning in Information Security (ConfML)

**Technical Content Analysis**

The "Sticker" in the title refers to Adversarial Patch Attacks—by placing carefully designed patterns in the physical world, AI vision systems can be made to produce incorrect identifications. For example, a specific sticker can cause autonomous driving systems to misidentify a stop sign as a speed limit sign.

The book's unique content organization adopts a narrative-driven structure:

**Historical Context**: Traces the development of adversarial machine learning from early academic research to modern commercial threats

**Technical Cases**:
- Adversarial attacks on image classifiers
- Attacks on speech recognition systems
- Manipulation of reinforcement learning systems
- Bypassing malware detectors

**Industry Perspective**: Based on hundreds of interviews covering academic researchers, policymakers, business leaders, and national security experts

**Expert Reviews**

- Miles Brundage (then Head of Policy Research at OpenAI): "Timely overview of this emerging risk landscape and what can be done about it."
- David Brumley (Carnegie Mellon University Professor): "This should be required reading to become AI/ML literate in the field."
- Nate Fick (former Endgame CEO): "Every leader and policymaker should read this compelling and persuasive book."

**Technical Characteristics**

1. **Strong Readability**: Core concepts understandable without deep mathematical background
2. **Industry Insights**: Provides industry practice perspectives difficult to obtain from academic papers
3. **Policy Relevance**: Connects technical issues with policy and governance concerns

**Limitations**

- Technical depth relatively limited; no detailed attack code implementations
- Focuses on conceptual understanding; practical operational guidance is limited
- Published in 2023; does not cover latest LLM-era threats

---

### 3. Evading EDR: The Definitive Guide to Defeating Endpoint Detection Systems

**Basic Information**

| Attribute | Details |
|-----------|---------|
| Author | Matt Hand |
| Publisher | No Starch Press |
| Publication Date | October 2023 |
| ISBN | 9781718503342 |

**Relevance to AI Security**

The reasons for including an Endpoint Detection and Response (EDR) book in an AI security reading list:

1. **AI Infrastructure Deployment Environment**: AI training clusters and inference servers run in EDR-protected endpoint environments
2. **APT Attack Paths**: Advanced Persistent Threats (APTs) attacking AI systems typically need to breach endpoint protection first
3. **Red Team Assessment Needs**: Assessing AI infrastructure security posture requires understanding endpoint security mechanisms
4. **Defense-in-Depth Design**: Designing AI system protection requires understanding underlying security layers

**Author Background**

Matt Hand is a Service Architect at SpecterOps, focusing on vulnerability research and EDR evasion techniques. He is responsible for improving the technical and execution capabilities of the Adversary Simulation team while serving as a subject matter expert on evasion tradecraft.

**Technical Content Architecture**

This book's technical depth ranks at the forefront among similar works:

**Chapters 1-5: EDR Architecture Analysis**
- EDR overall architecture (EDR-chitecture)
- Function-hooking DLL mechanisms
- Process/thread creation notifications
- Object notification mechanisms
- Image load and registry notifications

**Chapters 6-10: Critical Component Deep Dive**
- Filesystem minifilter drivers
- Network filter drivers
- Event Tracing for Windows (ETW)
- Scanner mechanisms
- Antimalware Scan Interface (AMSI)

**Chapters 11-12: Advanced Topics**
- Early Launch Antimalware Drivers (ELAM)
- Microsoft-Windows-Threat-Intelligence

**Chapter 13: Comprehensive Case Study**
- Complete "detection-aware" attack walkthrough

**Technical Characteristics**

1. **Windows Kernel Depth**: Deep coverage of Windows security subsystem internals
2. **Offense-Defense Perspective**: Each chapter analyzes detection mechanisms and provides evasion strategies
3. **Practice-Oriented**: Content derived from real red team experience

**Expert Reviews**

- Olaf Hartong (FalconForce Researcher): "A great book for red and blue [teams]!"
- Adam Chester (TrustedSec Red Teamer): "If you spend any time around EDRs, this book is an invaluable addition to your collection."

**Limitations**

- Focused on Windows platform; Linux/macOS coverage is limited
- Requires strong Windows kernel and systems programming background
- Does not directly discuss AI/ML technology; readers must make connections independently

---

### 4. Visual Threat Intelligence

**Basic Information**

| Attribute | Details |
|-----------|---------|
| Author | Thomas Roccia |
| Publisher | Security Break |
| Publication Date | 2023 |
| ISBN | 9780646879376 |

**Author Background**

Thomas Roccia is a Senior Security Researcher at Microsoft with over 12 years of cybersecurity industry experience. He previously worked on McAfee's Advanced Threat Research team and has extensive market and technical experience in threat intelligence. He also runs the SecurityBreak platform, showcasing latest projects and research findings.

**Relevance to AI Security**

Threat intelligence methodology has direct applicability to the AI security domain:

1. **APT Tracking**: Tracking advanced threat actors targeting AI systems
2. **Attack Attribution**: Analyzing sources of adversarial sample attacks, data poisoning events
3. **Trend Analysis**: Identifying evolution patterns of AI security threats
4. **Intelligence Sharing**: Exchanging AI security threat intelligence with industry partners

**Technical Content Architecture**

The book employs a visualization-driven methodology:

**Part One: Threat Intelligence Fundamentals**
- Intelligence types and lifecycle
- Analysis of Competing Hypotheses (ACH) framework
- Traffic Light Protocol (TLP) information sharing mechanism

**Part Two: Threat Actor Analysis**
- Diamond Model of Intrusion Analysis
- Tactics, Techniques, and Procedures (TTPs) analysis
- Attribution dilemma and handling methods

**Part Three: Tracking and Analysis Tools**
- Indicators of Compromise (IoC) and prioritization (Pyramid of Pain)
- YARA rule writing
- Sigma detection rules
- MSTICpy data analysis

**Part Four: Classic Case Studies**
- NotPetya deep analysis
- Shamoon attack attribution
- SolarWinds (Sunburst) supply chain attack

**Technical Characteristics**

1. **Visualization Methods**: Transforms complex threat intelligence into easily understandable graphical expressions
2. **Tool Practice**: Appendix contains comprehensive list of open-source threat intelligence tools
3. **Transferable Methodology**: Methods directly applicable to AI security threat tracking

**Expert Reviews**

- Jean-Pierre Lesueur (Phrozen Security Researcher): "This book is a great fundamentals refresher for experienced analysts, as well as an excellent entry material for newcomers."
- Kraven Security Review: "Marries cyber threat intelligence with visual storytelling so anyone can quickly understand complex abstract topics."

**Limitations**

- Not specifically targeting AI security threats; readers need to make knowledge transfers
- Visualization tools and techniques may require adjustment for specific needs
- Some cases are dated; recent AI-related threat cases need supplementation

---

## Comparative Analysis and Use Cases

### Technical Coverage Comparison

| Technical Domain | Adversarial AI | Sticker | Evading EDR | Visual TI |
|------------------|----------------|---------|-------------|-----------|
| Adversarial Sample Attacks | ★★★★★ | ★★★★☆ | ☆ | ☆ |
| Prompt Injection/LLM Security | ★★★★☆ | ★★☆☆☆ | ☆ | ☆ |
| Data Poisoning | ★★★★☆ | ★★★☆☆ | ☆ | ☆ |
| Model Extraction/Reverse | ★★★★☆ | ★★★☆☆ | ☆ | ☆ |
| MLSecOps | ★★★★★ | ★☆☆☆☆ | ☆ | ☆ |
| Endpoint Security Mechanisms | ★☆☆☆☆ | ☆ | ★★★★★ | ☆ |
| Threat Intelligence Analysis | ★★☆☆☆ | ★★☆☆☆ | ☆ | ★★★★★ |
| Code Implementation | ★★★★☆ | ★☆☆☆☆ | ★★★★☆ | ★★★☆☆ |

### Use Case Analysis

| Scenario | Recommended Book | Rationale |
|----------|------------------|-----------|
| AI System Threat Modeling | Adversarial AI Attacks | Provides complete AI threat modeling framework |
| AI Red Team Assessment | Adversarial AI + Evading EDR | Attack techniques + infrastructure penetration |
| AI System Defense Design | Adversarial AI Attacks | Covers defense strategies and MLSecOps |
| AI Security Concept Understanding | Not with a Bug | Strong readability, clear concepts |
| AI Threat Tracking | Visual Threat Intelligence | Threat intel methodology is transferable |
| Policy and Governance | Not with a Bug | Rich policy perspective |

### Reading Sequence Recommendations

Based on different backgrounds, the following reading sequences are provided as reference:

**Security Researchers**:
Adversarial AI Attacks → Evading EDR → Visual Threat Intelligence

**Machine Learning Engineers**:
Not with a Bug → Adversarial AI Attacks → Visual Threat Intelligence

**Security Architects**:
Adversarial AI Attacks → Visual Threat Intelligence → Evading EDR

---

## Companion Resources and Standards

### Code Repositories

| Book | Repository | Description |
|------|------------|-------------|
| Adversarial AI Attacks | [PacktPublishing/Adversarial-AI](https://github.com/PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies) | Attack demo code, defense implementations |

### Industry Standards and Frameworks

| Organization | Standard/Framework | Book Alignment |
|--------------|-------------------|----------------|
| OWASP | LLM Top 10 | Adversarial AI Attacks content aligned |
| OWASP | ML Top 10 | Machine learning security risk classification |
| NIST | AI Risk Management Framework | Risk management methodology reference |
| MITRE | ATLAS | AI adversarial threat knowledge base |

### Related Conferences and Communities

| Name | Type | Description |
|------|------|-------------|
| ConfML | Academic Conference | Conference on Applied ML in InfoSec (Hyrum Anderson co-founded) |
| DEFCON AI Village | Community | Annual DEFCON AI security track |
| Black Hat AI Summit | Summit | Black Hat AI security summit |

---

## Conclusions

### Technical Coverage Assessment

The four books together constitute a core knowledge framework for the AI security domain:

- **Adversarial AI Attacks**: Most comprehensive AI attack/defense technical guide, suitable for researchers needing deep understanding of attack techniques and defense strategies
- **Not with a Bug, But with a Sticker**: Best conceptual entry material, suitable for practitioners needing quick domain awareness
- **Evading EDR**: Essential reading for AI infrastructure underlying security, suitable for security personnel needing to assess or protect AI system runtime environments
- **Visual Threat Intelligence**: Threat intelligence methodology reference, suitable for intelligence analysts needing to track AI security threats

### Domain Development Trends

The AI security domain remains in rapid evolution. Books published in 2024-2025 have begun covering LLM security issues, but the field's rapid development means:

1. Book content needs to be combined with latest research papers and industry reports
2. Continuous updates from OWASP, NIST, MITRE and other organizations are important supplementary sources
3. Practical experience and case studies are crucial for understanding real threats

### Knowledge System Building

Building AI security knowledge systems requires multi-dimensional resource support:

- **Foundational Theory**: Books provide systematized knowledge frameworks
- **Cutting-Edge Research**: Academic papers and conferences provide latest technical advances
- **Industry Practice**: Standards frameworks and best practice guides provide implementation reference
- **Community Exchange**: Professional communities and conferences provide knowledge sharing platforms

---

## References

### Book Purchase Links

- [Amazon: Not with a Bug, But with a Sticker](https://www.amazon.com/Not-Bug-But-Sticker-Learning/dp/1119883989)
- [Packt: Adversarial AI Attacks, Mitigations, and Defense Strategies](https://www.packtpub.com/en-us/product/adversarial-ai-attacks-mitigations-and-defense-strategies-9781835088678)
- [No Starch Press: Evading EDR](https://nostarch.com/evading-edr)
- [Amazon: Visual Threat Intelligence](https://www.amazon.com/Visual-Threat-Intelligence-Illustrated-Researchers/dp/B0C7JCF8XD)

### Standards and Frameworks

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OWASP ML Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [MITRE ATLAS](https://atlas.mitre.org/)

### Industry Resources

- [Practical DevSecOps: Best AI Security Books](https://www.practical-devsecops.com/best-ai-security-books/)
- [GitHub: Adversarial AI Code Repository](https://github.com/PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies)

---

*This article was written by Innora Security Research Team. For questions or suggestions, please contact security@innora.ai*

*© 2026 Innora Security Research Team. Licensed under CC BY-NC 4.0*

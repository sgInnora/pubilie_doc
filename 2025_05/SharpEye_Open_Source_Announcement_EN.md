# SharpEye: Advanced Linux Intrusion Detection System Now Open Source

![SharpEye Logo](https://github.com/sgInnora/sharpeye/raw/main/assets/logo.png)

## Introducing SharpEye: A New Era in Linux Security

We are thrilled to announce that **SharpEye**, our advanced Linux intrusion detection and threat hunting system, is now available as an open-source project. After years of development and refinement, we at innora.ai have decided to share this powerful security tool with the global community to enhance the collective security posture of Linux environments worldwide.

**GitHub Repository**: [https://github.com/sgInnora/sharpeye](https://github.com/sgInnora/sharpeye)

### Why We Created SharpEye

In today's increasingly complex threat landscape, traditional security tools often fall short in detecting sophisticated attacks. Modern adversaries employ advanced techniques that can bypass conventional detection mechanisms, leaving systems vulnerable despite having security measures in place.

SharpEye was built to address these challenges by combining traditional system monitoring with cutting-edge machine learning and behavioral analysis. Our goal was to create a security solution that could:

1. Detect anomalous behavior rather than just matching known signatures
2. Provide comprehensive visibility across all aspects of a Linux system
3. Adapt to evolving threats without constant manual updates
4. Offer accessible security tools for organizations of all sizes

The result is SharpEye: a robust, intelligent, and adaptable security monitoring framework that provides deep visibility into Linux systems and can identify suspicious activities that other tools might miss.

## Comprehensive Feature Set

SharpEye distinguishes itself through a multifaceted approach to system security, monitoring various aspects of a Linux system to build a holistic security picture:

### System Resource Monitoring with Machine Learning

SharpEye continuously analyzes CPU, memory, and disk usage patterns to identify abnormal resource consumption that might indicate cryptominers, resource hijacking, or denial-of-service attacks. Our enhanced System Resources module combines traditional threshold-based monitoring with advanced machine learning capabilities:

- Uses Isolation Forest algorithm to detect anomalous resource patterns
- Identifies unusual CPU or memory spikes outside normal operational patterns
- Detects persistent high resource usage with unusual stability metrics
- Analyzes unexpected disk I/O patterns that may indicate data exfiltration
- Performs cross-resource correlation to detect sophisticated attack patterns
- Features time series analysis for identifying concerning trends
- Generates statistical models of normal behavior for comparison
- Implements self-training capability to adapt to your environment
- Monitors hidden processes and suspicious execution patterns
- Provides detailed metrics on resource usage anomalies with severity ratings

### User Account Security

Attackers often target user accounts as entry points or for privilege escalation. SharpEye provides comprehensive monitoring of user account activity:

- Detection of unauthorized account creation or modification
- Monitoring of privilege escalation attempts
- Analysis of suspicious login patterns, including unusual login times or locations
- Identification of dormant accounts showing sudden activity
- Detection of anomalous credential usage or password changes

### Process Analysis with Machine Learning

At the core of SharpEye is an advanced process monitoring system that employs both rule-based and machine learning approaches:

- Process lineage analysis to identify suspicious parent-child relationships
- Behavioral analysis to detect processes exhibiting unusual patterns
- Identification of processes attempting to hide their activity
- Memory scanning for signs of code injection or tampering
- Detection of known malicious process signatures and their variants
- Comprehensive process relationship mapping to identify attack chains
- Real-time process execution monitoring with anomaly detection
- Process environment and argument analysis for suspicious configurations
- Cross-correlation with network and file activities for holistic threat modeling
- Process integrity verification to detect runtime modifications

### ML-Based Cryptominer Detection

Unauthorized cryptocurrency mining is one of the most common forms of resource theft. SharpEye includes specialized machine learning algorithms specifically trained to detect cryptominers:

- CPU usage pattern analysis to identify mining algorithms
- Detection of mining traffic and communication patterns
- Identification of mining software signatures and behaviors
- Statistical analysis of process behavior compared to known mining patterns
- Correlation of multiple indicators for high-confidence detection

### Network Connection Monitoring

SharpEye carefully monitors network connections, providing visibility into:

- Unusual outbound connections to potentially malicious domains or IPs
- Detection of data exfiltration patterns
- Identification of command and control (C2) traffic
- Monitoring of unexpected listening ports or services
- Analysis of encrypted traffic patterns for anomalies

### Threat Intelligence Integration

SharpEye connects to multiple threat intelligence sources to enhance detection capabilities:

- Verification of network connections against known malicious IP databases
- Checking of file hashes against threat intelligence feeds
- Regular updates of threat signatures and indicators of compromise
- Customizable threat feed integration for specific environments
- Correlation of local findings with global threat landscape

### File System Integrity

Maintaining file system integrity is critical for security:

- Verification of system file integrity using cryptographic hashes
- Detection of unauthorized changes to critical system files
- Monitoring of sensitive configuration files
- Identification of hidden files or unusual permission changes
- Detection of potential backdoors or trojanized binaries

### Log Analysis Engine

SharpEye includes a sophisticated log analysis engine that:

- Monitors and analyzes system logs for suspicious patterns
- Correlates log events across different services
- Detects log tampering or deletion attempts
- Identifies authentication failures and brute force attempts
- Provides context-aware analysis of log anomalies

### Scheduled Task Inspection

Attackers often use scheduled tasks for persistence:

- Identifies suspicious cron jobs and scheduled tasks
- Detects modifications to existing scheduled tasks
- Monitors for unusual scheduling patterns
- Analyzes the content and behavior of scheduled scripts
- Correlates scheduled tasks with other system activities

### SSH Security and Advanced Analysis

SSH is a common target for attackers, and our comprehensive SSH Analyzer provides industry-leading detection capabilities:

- Monitors SSH configuration for security weaknesses and best practices
- Detects unauthorized access attempts and brute force attacks
- Identifies unusual SSH client configurations or key changes
- Monitors SSH session activities for anomalous behaviors
- Advanced SSH tunneling and port forwarding detection
- SSH key usage pattern analysis and anomaly detection
- Detection of keys used from multiple locations or during unusual hours
- Baseline creation and comparison for SSH configurations
- SSH key automation analysis (cron jobs, systemd services)
- Integration with threat intelligence for correlating suspicious access

### Kernel Module Analysis

The Linux kernel is a high-value target:

- Detection of malicious kernel modules and rootkits
- Verification of kernel module signatures and integrity
- Monitoring of kernel module loading and unloading
- Identification of signs of kernel memory tampering
- Analysis of kernel-level behavior anomalies

### Library Inspection

Dynamic libraries are often exploited:

- Identification of library hijacking attempts
- Detection of preloading attacks (LD_PRELOAD)
- Verification of library integrity and signatures
- Monitoring of library loading in sensitive processes
- Detection of unusual library dependencies

### Privilege Escalation Detection

SharpEye actively looks for signs of privilege escalation:

- Monitors for exploitable misconfigurations
- Detects exploitation of SUID/SGID binaries
- Identifies potential kernel exploits in use
- Monitors for exploitation of vulnerable services
- Detects suspicious capability changes

### Advanced Rootkit Detection

Our comprehensive Rootkit Detection module provides state-of-the-art capabilities:

- Kernel-level integrity verification for detecting low-level modifications
- Hidden process detection through multiple methods (syscall hooking, /proc comparison)
- Detection of hijacked interrupt handlers and service routines
- Network stack integrity verification to identify covert channels
- Hidden filesystem object detection (including overlay techniques)
- Memory-based rootkit detection (without persistent components)
- Cross-validation of system information through multiple query methods
- Anomaly detection for unexpected kernel behavior
- Runtime kernel integrity measurement and verification
- Behavioral analysis to detect rootkit evasion techniques

## The Significance of Open-Sourcing SharpEye

Making SharpEye open source represents our commitment to several core principles:

### Democratizing Security

By open-sourcing SharpEye, we aim to provide advanced security capabilities to organizations regardless of their size or resources. Security should not be a privilege of well-funded entities but a right for all. SharpEye brings enterprise-grade security monitoring to everyone, from individual Linux enthusiasts to large organizations.

### Transparency and Trust

Security tools must be trustworthy. By opening our code to public scrutiny, we're demonstrating our commitment to transparency. Users can verify exactly how SharpEye operates, identify potential privacy concerns, and confirm that the tool behaves as expected.

### Collaborative Innovation

Security is an ever-evolving challenge that benefits from diverse perspectives. By inviting the global community to contribute to SharpEye, we're enabling collective innovation that far exceeds what any single organization could achieve. Together, we can respond more effectively to emerging threats and develop new detection capabilities.

### Knowledge Sharing

The security community grows stronger when knowledge is shared. SharpEye not only provides a tool but serves as an educational platform for understanding Linux security, intrusion detection techniques, and machine learning applications in cybersecurity. Students, researchers, and professionals can all learn from and build upon our work.

## Getting Started with SharpEye

### System Requirements

SharpEye is designed to run efficiently on most Linux distributions with minimal resource overhead:

- **Operating System**: Linux-based operating system (Debian, Ubuntu, CentOS, RHEL, etc.)
- **Python**: Python 3.6 or higher
- **Permissions**: Root privileges for comprehensive scanning
- **Disk Space**: Minimal (~50MB for installation, variable log storage)
- **Memory**: At least 512MB RAM (1GB+ recommended)

### Installation

Installing SharpEye is straightforward:

```bash
# Clone the repository
git clone https://github.com/sgInnora/sharpeye.git

# Change to the SharpEye directory
cd sharpeye

# Run the installation script
sudo ./install.sh
```

The installation script will:
- Install necessary dependencies
- Set up required directories
- Configure basic settings
- Install SharpEye system services
- Create scheduled scans

### Basic Usage

SharpEye provides several modes of operation to suit different security needs:

#### Full System Scan

To run a comprehensive scan of your system:

```bash
sudo sharpeye --full-scan
```

This will activate all detection modules and generate a detailed report of potential security issues.

#### Targeted Module Scans

To run specific detection modules:

```bash
sudo sharpeye --module network
sudo sharpeye --module cryptominer
sudo sharpeye --module kernel
```

#### Baseline Comparison

SharpEye can establish a baseline of "normal" system behavior and compare future scans against it:

```bash
# Establish a baseline when system is in a known-good state
sudo sharpeye --establish-baseline

# Compare current state against the baseline
sudo sharpeye --compare-baseline
```

This approach is particularly effective for detecting subtle changes that might indicate a compromise.

#### Continuous Monitoring

For ongoing protection, SharpEye can be configured to run as a service:

```bash
# Start the SharpEye service
sudo systemctl start sharpeye

# Enable SharpEye to start at boot
sudo systemctl enable sharpeye
```

### Configuration

SharpEye is highly configurable to adapt to different environments:

- Configuration files are stored in `/etc/sharpeye/`
- The main configuration file is `config.yaml`
- Local overrides can be added to `local_config.yaml`
- Module-specific configurations are available in the config directory

Example configuration adjustments:

```yaml
# Set logging verbosity
general:
  log_level: "info"  # Options: debug, info, warning, error

# Configure scan frequency
scheduling:
  full_scan_interval: 86400  # seconds (daily)
  quick_scan_interval: 3600  # seconds (hourly)

# Adjust detection sensitivity
detection:
  sensitivity: "medium"  # Options: low, medium, high

# Enable/disable specific modules
modules:
  cryptominer:
    enabled: true
    sensitivity: "high"
  network:
    enabled: true
  kernel:
    enabled: true
```

## Current Development Status

As of May 2025, here is the current implementation status of SharpEye's core modules:

| Module | Status | Test Coverage |
|--------|--------|---------------|
| File System Integrity | ✅ Complete | 95% |
| Kernel Module Analysis | ✅ Complete | 94% |
| Library Inspection | ✅ Complete | 95% |
| Privilege Escalation Detection | ✅ Complete | 94% |
| Log Analysis Engine | ✅ Complete | 93% |
| Cryptominer Detection | ✅ Complete | 95% |
| System Resources | ✅ Complete | 100% |
| User Accounts | ✅ Complete | 100% |
| Processes | ✅ Complete | 100% |
| Network | ✅ Complete | 95% |
| Scheduled Tasks | ✅ Complete | 95% |
| SSH | ✅ Complete | 100% |
| Rootkit Detection | ✅ Complete | 100% |

The project now has a robust CI/CD pipeline with GitHub Actions, ensuring code quality and test coverage for all modules. As of our latest update (May 8, 2025), all 13 modules are fully implemented and comprehensively tested, with extensive documentation available in both English and Chinese. The CI/CD system includes automated testing with specialized handling for SQLite threading issues, comprehensive environment verification, and detailed diagnostic tools to ensure consistent quality across different environments.

## Future Development Roadmap

SharpEye is an evolving project with ambitious plans for future enhancements:

### Near-term Goals (6-12 months)

1. **Expanded OS Support**: Broaden compatibility to include more Linux distributions
2. **Enhanced UI**: Develop a comprehensive web interface for visualization and management
3. **API Enhancement**: Expand the API for better integration with SIEM and security orchestration tools
4. **Container Security**: Add specialized detection for container environments (Docker, Kubernetes)
5. **Cloud-Native Integration**: Develop plugins for major cloud platforms for seamless integration

### Mid-term Goals (12-24 months)

1. **Advanced AI Models**: Implement more sophisticated machine learning algorithms for behavior analysis
2. **Threat Hunting Playbooks**: Create automated workflows for common threat hunting scenarios
3. **Distributed Deployment**: Enhance capabilities for monitoring large-scale environments
4. **Real-time Correlation Engine**: Develop a real-time system to correlate events across multiple hosts
5. **Automated Response**: Add capabilities for automated threat mitigation and response

### Long-term Vision (2+ years)

1. **Predictive Security**: Move beyond detection to prediction of potential security issues
2. **Cross-Platform Support**: Extend core functionality to other operating systems
3. **Edge Computing Security**: Specialized modules for IoT and edge computing environments
4. **Industry-Specific Modules**: Develop tailored security modules for specific industries
5. **Security as Code Integration**: Seamless integration with infrastructure-as-code workflows

## How to Contribute

SharpEye thrives on community contributions. Here's how you can get involved:

### Contribution Areas

- **Code Contributions**: Enhance existing modules or develop new detection capabilities
- **Documentation**: Improve guides, examples, and technical documentation
- **Testing**: Help test SharpEye across different environments and scenarios
- **Bug Reports and Feature Requests**: Report issues and suggest improvements
- **Threat Intelligence**: Contribute to signature databases and detection rules
- **Translations**: Help make SharpEye accessible in more languages

### Getting Started with Contributing

1. **Fork the Repository**: Start by forking the SharpEye repository
2. **Set Up Development Environment**: Follow the development setup guide in the repository
3. **Pick an Issue**: Check the issue tracker for good first issues
4. **Make Changes**: Implement your enhancements or fixes
5. **Submit a Pull Request**: Contribute your changes back to the main project
6. **Join the Community**: Participate in discussions and help others

### Code Contribution Guidelines

- Follow the project's coding style and conventions
- Include tests for new functionality
- Ensure backward compatibility when possible
- Document your changes thoroughly
- Keep pull requests focused on a single issue or feature

### Recognition Program

We value all contributions and have implemented a recognition program:

- **Contributors List**: All contributors are acknowledged in the project
- **Maintainer Status**: Regular contributors may be invited to become maintainers
- **Feature Attribution**: Major contributions are credited in release notes
- **Community Spotlight**: Regular highlighting of exceptional contributions

## Community and Support

Join our growing community to get help, share ideas, and collaborate:

- **GitHub Discussions**: For questions, ideas, and general discussion
- **Issue Tracker**: For bug reports and feature requests
- **Documentation**: Comprehensive bilingual guides and reference materials in English and Chinese
- **Slack Channel**: For real-time collaboration and support
- **Monthly Webinars**: Covering new features, use cases, and best practices

### Documentation Resources

SharpEye features comprehensive bilingual documentation to support global users:

English Documentation:
- [User Guide](https://github.com/sgInnora/sharpeye/blob/main/docs/user_guide.md)
- [Module Reference](https://github.com/sgInnora/sharpeye/blob/main/docs/module_reference.md)
- [Machine Learning Analysis](https://github.com/sgInnora/sharpeye/blob/main/docs/machine_learning_analysis.md)
- [Testing Guide](https://github.com/sgInnora/sharpeye/blob/main/docs/testing.md)
- [Project Status](https://github.com/sgInnora/sharpeye/blob/main/docs/PROJECT_STATUS.md)
- [CI/CD Implementation Guide](https://github.com/sgInnora/sharpeye/blob/main/docs/CI_CD_STATUS.md)
- [Processes Module Documentation](https://github.com/sgInnora/sharpeye/blob/main/docs/modules/PROCESSES.md)
- [Rootkit Detector Documentation](https://github.com/sgInnora/sharpeye/blob/main/docs/modules/ROOTKIT_DETECTOR.md)

中文文档:
- [用户指南](https://github.com/sgInnora/sharpeye/blob/main/docs/user_guide_zh.md)
- [模块参考](https://github.com/sgInnora/sharpeye/blob/main/docs/module_reference_zh.md)
- [机器学习分析](https://github.com/sgInnora/sharpeye/blob/main/docs/machine_learning_analysis_zh.md)
- [测试指南](https://github.com/sgInnora/sharpeye/blob/main/docs/testing_zh.md)
- [项目状态](https://github.com/sgInnora/sharpeye/blob/main/docs/PROJECT_STATUS_ZH.md)
- [CI/CD实现指南](https://github.com/sgInnora/sharpeye/blob/main/docs/CI_CD_STATUS_ZH.md)
- [进程模块文档](https://github.com/sgInnora/sharpeye/blob/main/docs/modules/PROCESSES_ZH.md)
- [Rootkit检测器文档](https://github.com/sgInnora/sharpeye/blob/main/docs/modules/ROOTKIT_DETECTOR_ZH.md)

## About innora.ai

innora.ai specializes in developing advanced security solutions for modern computing environments. Our team combines expertise in malware analysis, threat intelligence, and machine learning to create cutting-edge security tools that help organizations protect their critical infrastructure.

By open-sourcing SharpEye, we're reaffirming our commitment to a more secure digital world through collaboration, innovation, and knowledge sharing.

---

## License

SharpEye is released under the MIT License, allowing for broad use, modification, and distribution while maintaining attribution to the original authors.

## Acknowledgments

- The innora.ai research team for their dedication in developing this tool
- All contributors and security researchers who have helped improve this project
- Open source security tools that have inspired aspects of SharpEye
- The Linux community for creating the foundation upon which this tool operates

---

Join us in building a more secure future for Linux systems worldwide. Together, we can stay one step ahead of evolving threats and protect our digital infrastructure.

Explore SharpEye today: [https://github.com/sgInnora/sharpeye](https://github.com/sgInnora/sharpeye)
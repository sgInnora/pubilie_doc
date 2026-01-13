---
name: integrating-cicd
description: Configures CI/CD pipelines for automated building, testing, and deployment. Supports GitHub Actions, GitLab CI, Jenkins, and cloud-native solutions. Triggers when user asks to "setup CI/CD", "create pipeline", "automate deployment", or "configure GitHub Actions".
---

# CI/CD Integration Skill

## Overview
Designs and implements continuous integration and deployment pipelines for automated software delivery, including build, test, security scanning, and deployment stages.

## Supported Platforms
- GitHub Actions
- GitLab CI/CD
- Jenkins
- Azure DevOps
- CircleCI
- AWS CodePipeline

## CI/CD Pipeline Design

```
┌─────────────────────────────────────────────────────────────┐
│                        CI/CD Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │  Build  │─▶│  Test   │─▶│Security │─▶│ Deploy  │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
│       │            │            │            │              │
│       ▼            ▼            ▼            ▼              │
│   Compile      Unit Tests    SAST/DAST   Staging/Prod      │
│   Package      Integration   Dependency   Blue/Green       │
│   Artifact     E2E Tests     License      Canary           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## GitHub Actions Templates

### Basic CI Workflow
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [18.x, 20.x]

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Run tests
        run: npm test -- --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}

  build-docker:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t app:${{ github.sha }} .

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push app:${{ github.sha }}
```

### Full CD Pipeline
```yaml
name: CD

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: npm test

  security-scan:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          severity: 'CRITICAL,HIGH'

      - name: Run SAST
        uses: github/codeql-action/analyze@v3

  deploy-staging:
    needs: security-scan
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to staging
        run: |
          # Deploy commands here
          echo "Deploying to staging..."

      - name: Run smoke tests
        run: npm run test:smoke

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to production
        run: |
          # Deploy commands here
          echo "Deploying to production..."

      - name: Health check
        run: curl -f https://api.example.com/health || exit 1
```

### Python CI Template
```yaml
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=xml

      - name: Type check with mypy
        run: |
          pip install mypy
          mypy src/
```

## GitLab CI Template

```yaml
stages:
  - build
  - test
  - security
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE

test:
  stage: test
  image: node:20
  script:
    - npm ci
    - npm test
  coverage: '/Coverage: \d+.\d+%/'

security_scan:
  stage: security
  image: aquasec/trivy:latest
  script:
    - trivy image --severity HIGH,CRITICAL $DOCKER_IMAGE

deploy_staging:
  stage: deploy
  environment:
    name: staging
    url: https://staging.example.com
  script:
    - kubectl apply -f k8s/staging/
  only:
    - develop

deploy_production:
  stage: deploy
  environment:
    name: production
    url: https://example.com
  script:
    - kubectl apply -f k8s/production/
  only:
    - main
  when: manual
```

## Jenkins Pipeline Template

```groovy
pipeline {
    agent any

    environment {
        DOCKER_REGISTRY = 'registry.example.com'
        APP_NAME = 'my-app'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build') {
            steps {
                sh 'npm ci'
                sh 'npm run build'
            }
        }

        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'npm run test:unit'
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh 'npm run test:integration'
                    }
                }
            }
        }

        stage('Security Scan') {
            steps {
                sh 'trivy fs --severity HIGH,CRITICAL .'
            }
        }

        stage('Docker Build') {
            steps {
                sh "docker build -t ${DOCKER_REGISTRY}/${APP_NAME}:${BUILD_NUMBER} ."
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                sh 'kubectl apply -f k8s/staging/'
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            input {
                message 'Deploy to production?'
            }
            steps {
                sh 'kubectl apply -f k8s/production/'
            }
        }
    }

    post {
        always {
            junit '**/test-results/*.xml'
        }
        failure {
            slackSend channel: '#deployments', message: "Build failed: ${BUILD_URL}"
        }
    }
}
```

## Best Practices Checklist

- [ ] Use matrix builds for multiple versions
- [ ] Cache dependencies for faster builds
- [ ] Run security scans in pipeline
- [ ] Use environment protection rules
- [ ] Implement rollback strategy
- [ ] Add health checks post-deploy
- [ ] Use secrets management
- [ ] Enable parallel jobs where possible
- [ ] Set up notifications for failures
- [ ] Document pipeline architecture

## Constraints
- Never commit secrets to repository
- Use environment-specific configurations
- Implement proper access controls
- Test pipeline changes in non-prod first
- Include manual approval for production
- Monitor pipeline performance metrics

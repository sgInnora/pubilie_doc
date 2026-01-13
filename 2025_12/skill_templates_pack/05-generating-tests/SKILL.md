---
name: generating-tests
description: Generates comprehensive test suites including unit tests, integration tests, and end-to-end tests. Supports multiple frameworks (Jest, Pytest, Go testing, JUnit). Achieves high code coverage with edge case handling. Triggers when user asks to "write tests", "generate test cases", "add unit tests", or "improve test coverage".
---

# Test Generation Skill

## Overview
Automatically generates high-quality test suites by analyzing code structure, identifying testable units, and creating comprehensive test cases with proper assertions.

## Supported Frameworks

| Language | Frameworks |
|----------|------------|
| JavaScript/TypeScript | Jest, Mocha, Vitest |
| Python | Pytest, Unittest |
| Go | testing, testify |
| Java | JUnit 5, TestNG |
| Rust | Built-in test framework |

## Test Categories

### Unit Tests
- Test individual functions/methods
- Mock external dependencies
- Focus on business logic

### Integration Tests
- Test component interactions
- Use real dependencies when feasible
- Test API endpoints

### End-to-End Tests
- Test complete user workflows
- Browser automation (Playwright, Cypress)
- Database state verification

## Test Generation Process

```
1. Code Analysis
   ├── Identify public interfaces
   ├── Extract input/output types
   ├── Detect dependencies to mock
   └── Find edge cases from logic

2. Test Case Design
   ├── Happy path scenarios
   ├── Error handling cases
   ├── Boundary conditions
   └── Edge cases

3. Test Implementation
   ├── Setup and teardown
   ├── Assertions
   ├── Mocking
   └── Test isolation

4. Coverage Check
   ├── Line coverage
   ├── Branch coverage
   └── Edge case coverage
```

## Templates

### Jest/TypeScript Template
```typescript
import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { FunctionUnderTest } from './module';
import { MockDependency } from './mocks';

describe('FunctionUnderTest', () => {
  let mockDep: jest.Mocked<MockDependency>;

  beforeEach(() => {
    mockDep = {
      method: jest.fn(),
    } as jest.Mocked<MockDependency>;
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('happy path', () => {
    it('should return expected result when given valid input', async () => {
      // Arrange
      const input = { /* valid input */ };
      const expected = { /* expected output */ };
      mockDep.method.mockResolvedValue(/* mock return */);

      // Act
      const result = await FunctionUnderTest(input, mockDep);

      // Assert
      expect(result).toEqual(expected);
      expect(mockDep.method).toHaveBeenCalledWith(/* expected args */);
    });
  });

  describe('error handling', () => {
    it('should throw error when input is invalid', async () => {
      // Arrange
      const invalidInput = { /* invalid input */ };

      // Act & Assert
      await expect(FunctionUnderTest(invalidInput, mockDep))
        .rejects
        .toThrow('Expected error message');
    });

    it('should handle dependency failure gracefully', async () => {
      // Arrange
      mockDep.method.mockRejectedValue(new Error('Dependency failed'));

      // Act & Assert
      await expect(FunctionUnderTest({ /* input */ }, mockDep))
        .rejects
        .toThrow('Dependency failed');
    });
  });

  describe('edge cases', () => {
    it('should handle empty input', async () => {
      const result = await FunctionUnderTest({}, mockDep);
      expect(result).toEqual(/* expected for empty */);
    });

    it('should handle maximum values', async () => {
      const input = { value: Number.MAX_SAFE_INTEGER };
      const result = await FunctionUnderTest(input, mockDep);
      expect(result).toBeDefined();
    });

    it('should handle null/undefined gracefully', async () => {
      await expect(FunctionUnderTest(null as any, mockDep))
        .rejects
        .toThrow();
    });
  });
});
```

### Pytest Template
```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from module import function_under_test

class TestFunctionUnderTest:
    """Test suite for function_under_test"""

    @pytest.fixture
    def mock_dependency(self):
        """Setup mock dependency"""
        mock = Mock()
        mock.method.return_value = "mock_result"
        return mock

    @pytest.fixture
    def sample_input(self):
        """Sample valid input"""
        return {"key": "value", "count": 10}

    # Happy Path Tests
    def test_returns_expected_result_with_valid_input(
        self, mock_dependency, sample_input
    ):
        """Should return expected result when given valid input"""
        # Act
        result = function_under_test(sample_input, mock_dependency)

        # Assert
        assert result is not None
        assert result["status"] == "success"
        mock_dependency.method.assert_called_once()

    # Error Handling Tests
    def test_raises_error_with_invalid_input(self, mock_dependency):
        """Should raise ValueError when input is invalid"""
        invalid_input = {"key": None}

        with pytest.raises(ValueError, match="Invalid input"):
            function_under_test(invalid_input, mock_dependency)

    def test_handles_dependency_failure(self, sample_input):
        """Should propagate dependency errors"""
        mock_dep = Mock()
        mock_dep.method.side_effect = ConnectionError("Failed")

        with pytest.raises(ConnectionError):
            function_under_test(sample_input, mock_dep)

    # Edge Cases
    @pytest.mark.parametrize("empty_input", [{}, [], None])
    def test_handles_empty_inputs(self, empty_input, mock_dependency):
        """Should handle various empty inputs"""
        with pytest.raises((ValueError, TypeError)):
            function_under_test(empty_input, mock_dependency)

    def test_handles_large_input(self, mock_dependency):
        """Should handle large input values"""
        large_input = {"data": "x" * 1_000_000}
        result = function_under_test(large_input, mock_dependency)
        assert result is not None

    # Async Tests
    @pytest.mark.asyncio
    async def test_async_operation(self, mock_dependency):
        """Should handle async operations correctly"""
        mock_dependency.async_method = AsyncMock(return_value="async_result")

        result = await async_function_under_test(mock_dependency)

        assert result == "expected"


# Fixtures for integration tests
@pytest.fixture(scope="module")
def database_connection():
    """Setup database connection for integration tests"""
    conn = create_test_database()
    yield conn
    conn.close()
    cleanup_test_database()
```

### Go Testing Template
```go
package module

import (
    "context"
    "errors"
    "testing"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
    "github.com/stretchr/testify/require"
)

// Mock setup
type MockDependency struct {
    mock.Mock
}

func (m *MockDependency) Method(ctx context.Context, input string) (string, error) {
    args := m.Called(ctx, input)
    return args.String(0), args.Error(1)
}

func TestFunctionUnderTest(t *testing.T) {
    t.Run("happy path", func(t *testing.T) {
        t.Run("returns expected result with valid input", func(t *testing.T) {
            // Arrange
            mockDep := new(MockDependency)
            mockDep.On("Method", mock.Anything, "input").Return("result", nil)

            // Act
            result, err := FunctionUnderTest(context.Background(), "input", mockDep)

            // Assert
            require.NoError(t, err)
            assert.Equal(t, "expected", result)
            mockDep.AssertExpectations(t)
        })
    })

    t.Run("error handling", func(t *testing.T) {
        t.Run("returns error when dependency fails", func(t *testing.T) {
            mockDep := new(MockDependency)
            mockDep.On("Method", mock.Anything, mock.Anything).
                Return("", errors.New("dependency error"))

            _, err := FunctionUnderTest(context.Background(), "input", mockDep)

            assert.Error(t, err)
            assert.Contains(t, err.Error(), "dependency error")
        })
    })

    t.Run("edge cases", func(t *testing.T) {
        t.Run("handles empty input", func(t *testing.T) {
            mockDep := new(MockDependency)

            _, err := FunctionUnderTest(context.Background(), "", mockDep)

            assert.Error(t, err)
        })
    })
}

// Table-driven tests
func TestFunctionWithMultipleInputs(t *testing.T) {
    tests := []struct {
        name     string
        input    string
        expected string
        wantErr  bool
    }{
        {"valid input", "test", "result", false},
        {"empty input", "", "", true},
        {"special chars", "!@#$", "sanitized", false},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result, err := ProcessInput(tt.input)

            if tt.wantErr {
                assert.Error(t, err)
                return
            }

            require.NoError(t, err)
            assert.Equal(t, tt.expected, result)
        })
    }
}
```

## Coverage Requirements
- Minimum 80% line coverage
- 100% coverage for critical paths
- All error branches tested
- Edge cases documented and tested

## Constraints
- Tests must be isolated (no shared state)
- Use descriptive test names
- Include arrange-act-assert comments
- Mock external dependencies
- Avoid testing implementation details

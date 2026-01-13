---
name: documenting-apis
description: Generates OpenAPI/Swagger documentation from code. Extracts endpoints, parameters, request/response schemas, and creates interactive API documentation. Supports REST, GraphQL, and gRPC. Triggers when user asks for "API docs", "generate swagger", "document endpoints", or "API reference".
---

# API Documentation Skill

## Overview
Automatically generates comprehensive API documentation by analyzing route definitions, extracting schemas, and producing OpenAPI 3.0 compliant specifications.

## Supported Formats
- OpenAPI 3.0/3.1 (default)
- Swagger 2.0
- GraphQL Schema
- AsyncAPI (for event-driven APIs)
- Markdown tables

## Documentation Process

```
1. Route Discovery
   ├── Scan controller/route files
   ├── Extract HTTP methods and paths
   └── Identify middleware and decorators

2. Schema Extraction
   ├── Parse request body types
   ├── Extract response models
   ├── Identify query/path parameters
   └── Map validation rules

3. Documentation Generation
   ├── Apply OpenAPI structure
   ├── Add examples and descriptions
   ├── Generate authentication docs
   └── Create error response schemas

4. Validation
   ├── Verify schema completeness
   ├── Check example validity
   └── Validate against OpenAPI spec
```

## OpenAPI Template

```yaml
openapi: 3.0.3
info:
  title: {API Title}
  description: |
    {Multi-line API description}

    ## Authentication
    {Authentication description}

    ## Rate Limiting
    {Rate limit info}
  version: {version}
  contact:
    name: API Support
    email: api-support@example.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://staging-api.example.com/v1
    description: Staging

tags:
  - name: {Resource1}
    description: {Resource1 operations}
  - name: {Resource2}
    description: {Resource2 operations}

paths:
  /{resource}:
    get:
      tags:
        - {Resource}
      summary: List all {resources}
      description: |
        Returns a paginated list of {resources}.

        Supports filtering by {filter_fields}.
      operationId: list{Resources}
      parameters:
        - name: page
          in: query
          description: Page number
          schema:
            type: integer
            default: 1
            minimum: 1
        - name: limit
          in: query
          description: Items per page
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/{Resource}List'
              example:
                data:
                  - id: "123"
                    name: "Example"
                pagination:
                  page: 1
                  total: 100
        '401':
          $ref: '#/components/responses/Unauthorized'
        '500':
          $ref: '#/components/responses/InternalError'
      security:
        - bearerAuth: []

    post:
      tags:
        - {Resource}
      summary: Create a {resource}
      description: Creates a new {resource} with the provided data.
      operationId: create{Resource}
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/{Resource}Create'
            example:
              name: "New Resource"
              description: "Description here"
      responses:
        '201':
          description: Resource created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/{Resource}'
        '400':
          $ref: '#/components/responses/BadRequest'
        '422':
          $ref: '#/components/responses/ValidationError'
      security:
        - bearerAuth: []

  /{resource}/{id}:
    get:
      tags:
        - {Resource}
      summary: Get a {resource}
      operationId: get{Resource}
      parameters:
        - name: id
          in: path
          required: true
          description: {Resource} ID
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/{Resource}'
        '404':
          $ref: '#/components/responses/NotFound'
      security:
        - bearerAuth: []

components:
  schemas:
    {Resource}:
      type: object
      required:
        - id
        - name
      properties:
        id:
          type: string
          format: uuid
          description: Unique identifier
          example: "550e8400-e29b-41d4-a716-446655440000"
        name:
          type: string
          description: Resource name
          minLength: 1
          maxLength: 255
          example: "Example Resource"
        createdAt:
          type: string
          format: date-time
          description: Creation timestamp
        updatedAt:
          type: string
          format: date-time
          description: Last update timestamp

    {Resource}Create:
      type: object
      required:
        - name
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 255

    Error:
      type: object
      properties:
        code:
          type: string
        message:
          type: string
        details:
          type: array
          items:
            type: object

  responses:
    BadRequest:
      description: Bad request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    ValidationError:
      description: Validation failed
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    InternalError:
      description: Internal server error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    apiKey:
      type: apiKey
      in: header
      name: X-API-Key
```

## Markdown Documentation Template

```markdown
# API Reference

## Base URL
`https://api.example.com/v1`

## Authentication
All API requests require authentication via Bearer token:
```
Authorization: Bearer <your_token>
```

## Endpoints

### {Resource}

#### List {Resources}
`GET /{resources}`

**Parameters**
| Name | Type | In | Description |
|------|------|-----|-------------|
| page | integer | query | Page number (default: 1) |
| limit | integer | query | Items per page (max: 100) |

**Response**
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "total": 100
  }
}
```

#### Create {Resource}
`POST /{resources}`

**Request Body**
```json
{
  "name": "string (required)",
  "description": "string (optional)"
}
```

**Response** `201 Created`
```json
{
  "id": "uuid",
  "name": "string",
  "createdAt": "datetime"
}
```

## Error Codes
| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |
```

## Constraints
- Include realistic examples for all schemas
- Document all error responses
- Add rate limiting information
- Include authentication details
- Version the API documentation

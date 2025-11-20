#!/bin/bash
set -e

# ===========================================
# Secure FastAPI Project Setup with uv
# ===========================================

PROJECT_NAME=${1:-"my-secure-api"}

echo "🚀 Creating secure FastAPI project: $PROJECT_NAME"

# Create project with uv
uv init $PROJECT_NAME
cd $PROJECT_NAME

# Add dependencies
echo "📦 Installing dependencies..."
uv add fastapi uvicorn[standard] pydantic pydantic-settings python-jose[cryptography] passlib[bcrypt] python-multipart

# Add dev dependencies
uv add --dev pytest pytest-cov pytest-asyncio httpx ruff bandit pre-commit

# Create project structure
echo "📁 Creating project structure..."
mkdir -p src/$PROJECT_NAME tests .github/workflows

# Create main app
cat > src/$PROJECT_NAME/__init__.py << 'EOF'
from .main import app

__all__ = ["app"]
EOF

cat > src/$PROJECT_NAME/main.py << 'EOF'
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel

app = FastAPI(
    title="Secure API",
    description="A security-first FastAPI application",
    version="1.0.0",
)

security = HTTPBearer()


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/protected")
async def protected_route(token: str = Depends(security)):
    """Example protected endpoint."""
    return {"message": "You have access!"}
EOF

# Create config with environment variables
cat > src/$PROJECT_NAME/config.py << 'EOF'
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    app_name: str = "Secure API"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    database_url: str = "sqlite:///./app.db"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
EOF

# Create tests
cat > tests/__init__.py << 'EOF'
EOF

cat > tests/test_main.py << 'EOF'
import pytest
from httpx import AsyncClient, ASGITransport

from src.${PROJECT_NAME//-/_}.main import app


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_protected_without_token(client):
    response = await client.get("/protected")
    assert response.status_code == 403
EOF

# Create .env.example
cat > .env.example << 'EOF'
APP_NAME=Secure API
DEBUG=false
SECRET_KEY=your-super-secret-key-change-this
DATABASE_URL=sqlite:///./app.db
ACCESS_TOKEN_EXPIRE_MINUTES=30
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
.venv/
.env
.env.*
!.env.example
.coverage
htmlcov/
.pytest_cache/
coverage.xml
.idea/
.vscode/
dist/
build/
*.egg-info/
*.pem
*.key
secrets/
EOF

# Create pre-commit config
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-added-large-files

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.4
    hooks:
      - id: gitleaks
EOF

# Create ruff config
cat > ruff.toml << 'EOF'
line-length = 88
target-version = "py312"

[lint]
select = ["E", "W", "F", "I", "B", "S", "C4", "UP"]
ignore = ["S101"]

[lint.per-file-ignores]
"tests/*" = ["S101"]
EOF

# Create GitHub Actions workflow
mkdir -p .github/workflows

cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read
  security-events: write

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run pytest --cov=src --cov-report=xml -v
      - uses: codecov/codecov-action@v4
        if: always()

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run bandit -r src/ -ll
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'
EOF

# Create README
cat > README.md << EOF
# $PROJECT_NAME

A secure FastAPI application with CI/CD pipeline.

## Setup

\`\`\`bash
cp .env.example .env
uv sync
uv run pre-commit install
uv run uvicorn src.${PROJECT_NAME//-/_}.main:app --reload
\`\`\`

## Testing

\`\`\`bash
uv run pytest --cov=src
\`\`\`

## API Docs

- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
EOF

# Initialize git
git init
git add .
git commit -m "Initial commit: Secure FastAPI project"

# Install pre-commit hooks
uv run pre-commit install

echo ""
echo "✅ Project setup complete!"
echo ""
echo "Next steps:"
echo "  cd $PROJECT_NAME"
echo "  cp .env.example .env"
echo "  uv run uvicorn src.${PROJECT_NAME//-/_}.main:app --reload"

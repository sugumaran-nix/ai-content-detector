# Security and Vulnerability Audit

**Repository:** `sugumaran-nix/ai-content-detector`  
**Audit date:** 2026-08-15  
**Scope:** Frontend (`index.html`, `ui-state.js`), FastAPI backend, model/data build scripts, dependency manifests, Dockerfiles, Render configuration, and tracked artifacts.

## Executive summary

The repository is a small static frontend plus a FastAPI inference service. The audit found no committed private-key or credential material in the source/configuration files scanned. The most important risks were supply-chain and operational rather than classic injection vulnerabilities: an outdated development-only pytest constraint, unsafe deserialization of trusted pickle artifacts, unpinned training-data downloads, permissive local-development CORS defaults, and internal error/request metadata handling.

The audit remediated the runtime-facing issues that could be addressed without changing the model format. The backend now sanitizes client-supplied request IDs, adds baseline response security headers, returns generic public inference errors, rejects unsafe blank/oversized batch inputs, and excludes the known vulnerable pytest release line. Complete integration and residual-risk guidance is documented below.

> This is a repository-level engineering audit, not a guarantee that a deployed service is free of vulnerabilities. Production infrastructure, DNS, TLS termination, hosting IAM, secrets, network policy, and monitoring must also be reviewed.

## Methodology

The review combined source inspection, repository/configuration inventory, targeted secret-pattern scanning, Python compilation, backend unit/API tests, Bandit static analysis, and `pip-audit` dependency analysis. Dependency vulnerability scanning follows the purpose of `pip-audit`, which reports known vulnerabilities from Python package advisory sources.[1] API risks were evaluated against the OWASP API Security Top 10 categories, especially unrestricted resource consumption, security misconfiguration, unsafe consumption of APIs, and improper inventory management.[2]

## Findings and status

| ID | Severity | Area | Finding | Status |
|---|---|---|---|---|
| SEC-001 | Medium | Dependency | `pytest>=8,<9` resolved to a release reported by `pip-audit` as `PYSEC-2026-1845`, fixed in `9.0.3`. | **Fixed** by changing the requirement to `pytest>=9.0.3,<10.0`. |
| SEC-002 | Medium | Deserialization | `pickle.load` is used for the classifier and reference language model. Pickle is code-executing serialization when the input is untrusted. | **Accepted with containment.** Artifacts are repository build outputs and must only come from a trusted build pipeline. Migrating to a safer artifact format is recommended as a future hardening project. |
| SEC-003 | Medium | Supply chain | Training/build scripts call `load_dataset("Hello-SimpleAI/HC3", ...)` without pinning a dataset revision. A future upstream change could alter training inputs or build outputs. | **Open.** Pin a reviewed dataset commit/revision and record checksums in a reproducible training manifest before rebuilding production artifacts. |
| SEC-004 | Medium | Configuration | `ALLOWED_ORIGINS` defaults to `*` for local development. If deployed without an override, any origin can make browser requests to the API. | **Mitigated by deployment config; open as a safe-default concern.** Production must set exact origins and deployment checks should fail when the wildcard is used outside development. FastAPI documents explicit CORS origin configuration for browser clients.[3] |
| SEC-005 | Medium | Information disclosure | Inference exceptions previously returned `str(exc)`, potentially exposing filesystem paths or dependency details. | **Fixed** with generic `500`/`503` response details and server-side logging. |
| SEC-006 | Low | Header injection/trace integrity | Client-provided `X-Request-ID` was echoed without validation. | **Fixed** with a 64-character allowlist and generated fallback IDs. |
| SEC-007 | Low | Browser hardening | The API did not set common response hardening headers. | **Fixed** with `nosniff`, `DENY`, `no-referrer`, and a restrictive `Permissions-Policy`. |
| SEC-008 | Low | Resource consumption | The API has a per-IP in-process sliding-window limiter and bounded input/batch sizes. | **Mitigated, residual risk remains.** Multi-worker/multi-replica deployments need a gateway or shared limiter because the current store is process-local. |
| SEC-009 | Low | Frontend DOM sinks | The frontend uses `innerHTML` for controlled UI templates. The audit found no direct remote HTML injection path; dynamic user text is escaped where inserted into history/annotations. | **Monitored.** Prefer DOM node construction or Trusted Types if future features introduce remote/user-controlled HTML. |
| SEC-010 | Low | Training scripts | Bandit flagged standard `random` use in dataset sampling and generation scripts. | **Accepted.** These calls are for deterministic dataset sampling, not secrets, tokens, or security decisions. |

## Detailed review by attack surface

### API input and resource controls

All document inputs are trimmed and validated. Blank strings, non-string values, oversized items, and invalid batch shapes are rejected before inference. The API also enforces minimum word counts for document endpoints, caps batch size, and applies a POST rate limit. These controls address the primary denial-of-service concern for an inference service, although deployment-level concurrency and timeout controls should still be configured.

### Error handling and observability

Public inference failures now return stable messages: `Inference failed.` or `Model service is not ready.` Detailed exceptions are logged server-side. Every response carries a bounded request ID and processing-time header, enabling correlation without echoing arbitrary header content.

### CORS and deployment configuration

The Render configuration sets a specific production frontend origin. Local development may use the wildcard fallback, but production deployment must override it. A CI/deployment policy should reject `ALLOWED_ORIGINS=*` for production environments. CORS is not authentication; if the API becomes private, add an authentication layer and authorization checks rather than relying on origin filtering.

### Serialization and model artifacts

The backend loads `classifier.pkl` and `reference_lm.pkl`. These files are executable Python serialization and must be treated as trusted code-adjacent artifacts. Do not accept model files from user uploads, remote URLs, or unreviewed pull requests into a production build. Recommended future migration options include a safer data-only format for the reference LM and a model format with explicit safe-loading semantics for the classifier, plus artifact hashes checked during image builds.

### External training data

The runtime service does not fetch arbitrary user-supplied URLs. The build scripts do fetch the HC3 dataset from Hugging Face. Because the dataset revision is not pinned, training is not fully reproducible. Pin a specific revision, verify the downloaded dataset hash or manifest, and separate training credentials/network access from the runtime image build where possible.

### Frontend security

The frontend is a static document with no backend secret material. It imports a public Transformers runtime from jsDelivr and a public model from Hugging Face. The browser uses local/session storage for theme/history state. Do not place private API keys in this file. If the frontend is later served with a strict CSP, the current inline styles/scripts will need to be moved into external assets or explicitly nonce/hash authorized.

## Verification performed

The following checks were run during the audit:

```bash
PYTHONPATH=backend pytest -q backend/tests
python3 -m compileall -q backend
node --check /tmp/ai-detector.js
node --check ui-state.js
git diff --check
pip-audit -r backend/requirements.txt
bandit -r backend -x backend/tests
```

Before remediation, `pip-audit` reported one known vulnerability in the old pytest constraint. After the requirements change, rerun the command in a clean environment and treat any non-zero result as a release blocker unless the finding is explicitly reviewed and documented.

## Recommended next steps

The highest-priority remaining item is to pin and attest the Hugging Face training dataset revision before the next model rebuild. The next operational priority is enforcing exact CORS origins and a shared rate limiter at the edge for multi-worker or multi-replica deployments. The longer-term model-security priority is replacing pickle artifacts with safer, integrity-checked formats.

## References

[1]: https://pypi.org/project/pip-audit/ "pip-audit: scanning Python environments for known vulnerabilities"

[2]: https://owasp.org/API-Security/editions/2023/en/0x11-t10/ "OWASP API Security Top 10 2023"

[3]: https://fastapi.tiangolo.com/tutorial/cors/ "FastAPI CORS configuration"

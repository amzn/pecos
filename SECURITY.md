# Security Policy

## Reporting a Vulnerability

If you discover a potential security issue in this project, please do **not** create a
public GitHub issue. Instead, report it to AWS/Amazon Security via the
[AWS Vulnerability Reporting page](https://aws.amazon.com/security/vulnerability-reporting/)
or email aws-security@amazon.com. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Loading models and preprocessors from trusted sources only

PECOS serializes some model and preprocessor artifacts with Python's `pickle` module
(for example, the `SklearnTfidf` and `SklearnHashing` vectorizers persist a
`vectorizer.pkl`). **Deserializing a pickle file can execute arbitrary code.**

Only load model, vectorizer, or preprocessor folders that come from a source you fully
trust and control. Do **not** load artifacts whose path or contents can be influenced by
an untrusted party — for example on shared or multi-tenant infrastructure, from shared
caches, or from artifact paths derived from external input. This is standard practice for
pickle-backed ML artifacts; PECOS is designed for trusted local/batch workflows.

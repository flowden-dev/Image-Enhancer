# Security

Do not post API keys, private images or full API responses in public issues.
Use GitHub's private vulnerability reporting if enabled for the repository;
otherwise contact a repository maintainer privately before disclosing details.

The application sends an image, prompt and optional mask to OpenAI only after
confirmation. Keys are kept in process memory or read from `OPENAI_API_KEY`.
The app disables automatic retries, redirects and environment proxy overrides.
Sources are never overwritten; native output is verified before publication.

Image parsers have file/pixel bounds. Treat images from unknown sources with care
and keep dependencies current. Supported-format limits are in [formats](docs/formats.md).

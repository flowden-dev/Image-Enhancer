# Repository instructions

Read the README and the relevant modules and tests before editing. Keep changes focused.

## Invariants

- Never overwrite sources, follow input links in recursive scans, or retry paid requests automatically.
- At most one explicit image request per new AI result. Identical-image reuse makes no request. No hidden OCR, probes or fallback models.
- No live API calls without explicit authorization. Never log keys, prompts or exact wording.
- All Tk operations stay on the main thread; workers communicate through queues.
- Protect regions override translate regions. Restore source pixels locally before native encoding.
- Preserve supported source descriptors for matching formats. Preflight the chosen output encoder before spending credit.
- Use explicit integer widths before NumPy multiplication/shifts; verify alpha and color semantics.
- Keep batch memory bounded to the active image; preserve recovery files before risky postprocessing.

## Workflow

Follow the [contribution guide](CONTRIBUTING.md) for lint, full desktop tests and package builds.
Use synthetic image fixtures and update the documentation when behavior changes.
Keep publication and live API requests separate from offline validation.

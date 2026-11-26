"""Shared, user-facing error type."""


class EnhancerError(Exception):
    """An error safe to report without a traceback containing request data."""

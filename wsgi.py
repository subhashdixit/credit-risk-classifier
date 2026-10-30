"""Gunicorn and local Flask entry point."""

from credit_risk.web import create_app

app = create_app()

# zettlab-overlay(connector-intent): keep the legacy API while adapter owns validation; upstream: none
"""Compatibility imports for Zettlab connector setup proposals."""

from gateway.platforms.zet_agent_connector_setup_intent import (
    CONNECTOR_SETUP_SCHEMA as CONNECTOR_SETUP_SCHEMA,
    connector_setup_result as connector_setup_result,
    normalize_connector_setup as normalize_connector_setup,
)

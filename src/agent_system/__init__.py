"""Pluggable multi-agent search experiments (protocol ``agent_v1``).

The frozen reproduction pipeline in :mod:`src.scaling_model` stays untouched;
this package re-implements the same synchronous protocol behind a policy
interface so rule-based, LLM and replay decision makers share one runner.
"""

PROTOCOL_VERSION = "agent_v1"

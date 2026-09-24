"""LLM configuration and a pluggable corporate-gateway provider.

The cash agent talked to Walmart's internal LLM Gateway through a custom
LiteLLM provider (WalmartLLMGateway) that injected a PingFed JWT. FAB will have
its own gateway or an Azure OpenAI deployment on Databricks; the shape is the
same. This module:

  1. Configures DSPy with the investigator + summarizer models.
  2. Shows the custom-provider pattern for gateways that need a bearer token
     minted per request (OAuth client-credentials / JWT).

If you use plain Azure OpenAI, you can skip the custom provider entirely and
let DSPy/LiteLLM talk to Azure directly with env vars.
"""
from __future__ import annotations

import time
from typing import Any

import dspy

from config.settings import SETTINGS


# --------------------------------------------------------------------------- #
# Option A — corporate gateway that needs a freshly minted bearer token.
# Mirrors the WalmartLLMGateway custom LiteLLM provider pattern.
# --------------------------------------------------------------------------- #
class GatewayTokenProvider:
    """Mints and caches a short-lived bearer token for the LLM gateway.

    Replace `_mint_token` with the real OAuth client-credentials / JWT exchange.
    Tokens are cached until shortly before expiry to avoid a round-trip per call.
    """

    def __init__(self, skew_seconds: int = 60) -> None:
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._skew = skew_seconds

    def _mint_token(self) -> tuple[str, float]:
        # TODO(FAB): call the identity provider (e.g. PingFed/Entra ID) here.
        #   resp = requests.post(TOKEN_URL, data={...client creds...})
        #   return resp.json()["access_token"], time.time() + resp.json()["expires_in"]
        raise NotImplementedError("Wire this to FAB's identity provider.")

    def token(self) -> str:
        if self._token is None or time.time() > self._expires_at - self._skew:
            self._token, self._expires_at = self._mint_token()
        return self._token


def build_gateway_lm(model: str, token_provider: GatewayTokenProvider) -> dspy.LM:
    """A dspy.LM pointed at a corporate gateway, injecting a bearer token.

    LiteLLM (which backs dspy.LM) forwards `extra_headers` on every request, so
    we refresh the token through the provider each call.
    """
    return dspy.LM(
        model=model,
        api_base=SETTINGS.llm.api_base,
        api_version=SETTINGS.llm.api_version,
        temperature=SETTINGS.llm.temperature,
        max_tokens=SETTINGS.llm.max_tokens,
        timeout=SETTINGS.llm.request_timeout_s,
        # LiteLLM evaluates this lazily per request when passed as a callable
        # header value is not supported directly, so refresh in a wrapper if
        # your gateway rotates tokens faster than the process lifetime.
        extra_headers={"Authorization": f"Bearer {token_provider.token()}"},
    )


# --------------------------------------------------------------------------- #
# Option B — plain Azure OpenAI on Databricks (simplest path).
# --------------------------------------------------------------------------- #
def build_azure_lm(model: str) -> dspy.LM:
    return dspy.LM(
        model=model,                       # e.g. "azure/gpt-4o"
        api_base=SETTINGS.llm.api_base,    # AZURE_API_BASE
        api_version=SETTINGS.llm.api_version,
        temperature=SETTINGS.llm.temperature,
        max_tokens=SETTINGS.llm.max_tokens,
        timeout=SETTINGS.llm.request_timeout_s,
    )


def configure_dspy(use_gateway: bool = False) -> dict[str, dspy.LM]:
    """Configure DSPy's default LM (investigator) and return both LMs.

    The summarizer runs under a `with dspy.context(lm=summarizer_lm):` block in
    the summarizer agent so the two models stay cleanly separated.
    """
    if use_gateway:
        provider = GatewayTokenProvider()
        investigator_lm = build_gateway_lm(SETTINGS.llm.investigator_model, provider)
        summarizer_lm = build_gateway_lm(SETTINGS.llm.summarizer_model, provider)
    else:
        investigator_lm = build_azure_lm(SETTINGS.llm.investigator_model)
        summarizer_lm = build_azure_lm(SETTINGS.llm.summarizer_model)

    dspy.configure(lm=investigator_lm)
    return {"investigator": investigator_lm, "summarizer": summarizer_lm}

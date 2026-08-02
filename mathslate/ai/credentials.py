"""Private, OS-backed storage for optional AI credentials.

Keys never belong in notebooks, project files, or MathSlate's configuration.
When the user explicitly asks MathSlate to remember one, ``keyring`` routes it
to Windows Credential Manager, macOS Keychain, or the desktop secret service.
The module is lazy and optional so importing :mod:`mathslate.ai` remains cheap.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

from ..errors import UnsupportedInputError

__all__ = [
    "SavedCredential",
    "credential_store_available",
    "load_credential",
    "save_credential",
    "delete_credential",
]

_SERVICE = "mathslate.ai"
_DEFAULT_PROVIDER = "__default_provider__"


@dataclass(frozen=True)
class SavedCredential:
    provider: str
    api_key: str
    model: str | None = None


def _keyring(*, required: bool):
    if sys.platform == "emscripten":
        if required:
            raise UnsupportedInputError(
                "persistent API-key storage is unavailable in JupyterLite. "
                "Use a local Jupyter kernel or keep the key for this session only."
            )
        return None
    try:
        module = importlib.import_module("keyring")
        backend = module.get_keyring()
        if float(getattr(backend, "priority", 0)) <= 0:
            raise RuntimeError("no usable OS credential backend")
        return module
    except Exception as exc:
        if required:
            raise UnsupportedInputError(
                "secure credential storage is unavailable. Install `keyring` "
                "and make sure the operating-system credential service is enabled."
            ) from exc
        return None


def credential_store_available() -> bool:
    """Whether this local Python session has a usable secure key store."""
    return _keyring(required=False) is not None


def save_credential(provider: str, api_key: str, model: str | None = None) -> None:
    """Remember one provider credential without writing it to the project."""
    if not api_key:
        raise UnsupportedInputError("an empty API key cannot be saved.")
    keyring = _keyring(required=True)
    assert keyring is not None
    try:
        keyring.set_password(_SERVICE, f"provider:{provider}", api_key)
        keyring.set_password(_SERVICE, _DEFAULT_PROVIDER, provider)
        if model:
            keyring.set_password(_SERVICE, f"model:{provider}", model)
    except Exception as exc:
        raise UnsupportedInputError(
            "the operating-system credential store refused the API key. "
            "Use session-only mode or check the credential-service settings."
        ) from exc


def load_credential(provider: str | None = None) -> SavedCredential | None:
    """Load the named or most recently saved provider, without exposing it."""
    keyring = _keyring(required=False)
    if keyring is None:
        return None
    try:
        chosen = provider or keyring.get_password(_SERVICE, _DEFAULT_PROVIDER)
        if not chosen:
            return None
        key = keyring.get_password(_SERVICE, f"provider:{chosen}")
        if not key:
            return None
        model = keyring.get_password(_SERVICE, f"model:{chosen}")
        return SavedCredential(chosen, key, model or None)
    except Exception:
        # Automatic discovery must not break ordinary offline MathSlate use.
        # Explicit save/delete operations do report credential-store failures.
        return None


def delete_credential(provider: str | None = None) -> None:
    """Remove a saved key (and its model hint) from the OS credential store."""
    keyring = _keyring(required=True)
    assert keyring is not None
    try:
        chosen = provider or keyring.get_password(_SERVICE, _DEFAULT_PROVIDER)
        if not chosen:
            return
        for account in (f"provider:{chosen}", f"model:{chosen}"):
            try:
                keyring.delete_password(_SERVICE, account)
            except Exception:
                pass
        default = keyring.get_password(_SERVICE, _DEFAULT_PROVIDER)
        if default == chosen:
            try:
                keyring.delete_password(_SERVICE, _DEFAULT_PROVIDER)
            except Exception:
                pass
    except Exception as exc:
        raise UnsupportedInputError(
            "the operating-system credential store could not remove the saved key."
        ) from exc

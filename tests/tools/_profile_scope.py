"""Helpers for asserting profile-secret-scope behavior (shared gateway mode).

Behavior-driven by design: the caller defines ONE mapping with every secret its
module needs, and these helpers purge or poison ``os.environ`` by iterating
that mapping — never by naming individual variables. A test built on them keeps
failing when the secret set changes or when an implementation read bypasses the
scope, instead of silently passing on hardcoded names.
"""

import contextlib


@contextlib.contextmanager
def mux_profile_scope(monkeypatch, scope, poison_environ=False):
    """Activate multiplexing with *scope* installed as the profile secrets.

    Every key in *scope* is removed from ``os.environ`` (or, with
    ``poison_environ=True``, replaced by a ``stale-<NAME>`` decoy). Either way
    a read that bypasses ``get_secret`` cannot see the scoped value: it gets
    nothing, or a decoy the test can assert never surfaces.
    """
    from agent import secret_scope

    for name in scope:
        if poison_environ:
            monkeypatch.setenv(name, f"stale-{name}")
        else:
            monkeypatch.delenv(name, raising=False)
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(dict(scope))
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous)


def request_fingerprint(req):
    """Serialize a urllib Request (URL + headers + body) for leak assertions."""
    body = req.data.decode("utf-8", errors="replace") if req.data else ""
    import json

    return "\n".join([req.full_url, json.dumps(dict(req.headers)), body])

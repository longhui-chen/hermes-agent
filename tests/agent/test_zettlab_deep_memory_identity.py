from gateway.deep_memory_identity import bounded_identity_header


def test_deep_memory_identity_header_is_bounded():
    assert bounded_identity_header("  iam:issuer:user:alice  ") == (
        "iam:issuer:user:alice"
    )
    assert bounded_identity_header("bad\nidentity") == ""
    assert bounded_identity_header("x" * 1025) == ""

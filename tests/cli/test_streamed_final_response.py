from cli import _should_render_final_response_panel


def test_transformed_response_renders_after_token_stream():
    assert _should_render_final_response_panel(
        response_transformed=True,
        token_streamed=True,
        tts_streamed=False,
    )


def test_transformed_response_renders_after_tts_stream():
    assert _should_render_final_response_panel(
        response_transformed=True,
        token_streamed=False,
        tts_streamed=True,
    )


def test_unmodified_streamed_response_does_not_render_twice():
    assert not _should_render_final_response_panel(
        response_transformed=False,
        token_streamed=True,
        tts_streamed=False,
    )

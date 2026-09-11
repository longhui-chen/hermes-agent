import pytest

from gateway.platforms.connector_input_guard import requests_secret_input


@pytest.mark.parametrize('question', [
    '请在安全连接卡片的受保护输入框中填写新 PAT 并保存，不要把凭据发送到聊天中。',
    '请输入 API Token',
    '请在安全连接卡中完成公司 Jira 的连接配置。',
    'Complete the setup in the secure connection card.',
    'Paste your API key in the secure card',
    'Enter the camera password',
    '请提供打印机的 LAN 访问码',
])
def test_rejects_secret_solicitation_even_when_labelled_secure(question):
    assert requests_secret_input(question)


@pytest.mark.parametrize('question', [
    'Which authentication method: PAT or OAuth?',
    '请选择认证方式',
    'What is the token expiration date?',
    'Which file should I open?',
    'Connector setup',
])
def test_preserves_non_secret_clarification(question):
    assert not requests_secret_input(question)

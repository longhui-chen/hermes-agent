import pytest

from tools import terminal_tool


@pytest.mark.parametrize(
    "args",
    [
        ["system", "status"],
        ["system", "smart-info", "--device", "/dev/sda"],
        ["file", "list", "--path", ".", "--limit", "20"],
        ["file", "write", "--path", "notes/a.txt", "--stdin", "--parents"],
        ["file", "copy", "--source", "a.txt", "--target", "archive/a.txt"],
        ["file", "search", "--query", "report"],
    ],
)
def test_agentcomputer_cli_parser_accepts_allowlisted_commands(args):
    terminal_tool._validate_agentcomputer_cli_args(args)


@pytest.mark.parametrize(
    "args",
    [
        ["system", "shutdown"],
        ["system", "time", "--set", "now"],
        ["file", "read", "--path", "/etc/passwd"],
        ["file", "read", "--path", "../secret"],
        ["file", "write", "--path", "a.txt", "--content", "secret"],
        ["file", "write", "--path", "a.txt"],
        ["file", "delete", "--path", "a.txt", "--recursive"],
    ],
)
def test_agentcomputer_cli_parser_rejects_mutations_and_escape(args):
    with pytest.raises(ValueError):
        terminal_tool._validate_agentcomputer_cli_args(args)

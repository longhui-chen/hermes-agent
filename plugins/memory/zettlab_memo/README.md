# Zettlab Memo Memory Provider

Managed on-device provider for Zettlab Memo. It complements, rather than
replaces, Hermes' built-in `MEMORY.md` and `USER.md` memory.

The Local Server writes `memory.provider: zettlab_memo` and provisions the
loopback URL plus action token. The provider then:

- prefetches relevant structured graph facts before a turn;
- mirrors successful native Hermes memory writes for structured extraction;
- leaves explicit `memo_write`, `memo_recall`, and `memo_confirm` operations on
  the managed MCP transport to avoid duplicate tool schemas.

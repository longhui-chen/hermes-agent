# Invoking Bundled Scripts

This file lives under `references/` on purpose. Template substitution runs over
SKILL.md bodies, so the placeholders taught here would be rewritten into *this*
package's absolute path before you ever saw them — and the examples would teach
a path that only exists on one machine. Supporting files are served verbatim,
which is the only place the syntax can be shown as syntax.

## The rule

Never write a relative path to a script the skill ships. The working directory a
command runs in is **not** the skill package — on Zettlab devices it is the
agent's own output directory, so `python scripts/x.py` cannot find the script at
all. Relative paths only ever worked when the model happened to convert them
using the injected skill directory, which is a soft, per-model behaviour.

Use the `${HERMES_SKILL_DIR}` template variable. It is substituted with the
package's absolute path when SKILL.md loads:

```
python3 "${HERMES_SKILL_DIR}/scripts/search.py" --query "..."
bash "${HERMES_SKILL_DIR}/scripts/setup.sh"
```

- Spell it exactly `${HERMES_SKILL_DIR}` — a bare `SKILL_DIR` or a missing brace
  is not substituted and fails silently as a literal path.
- Quote the expansion. The substituted path may contain spaces.
- Prefer `python3` over `python`; the latter is absent on some device images.
- Only for scripts **this package ships**. Paths into a cloned repository, the
  user's project, or `/tmp` are not skill paths — leave those relative to
  whatever the surrounding steps establish.

## Inputs and outputs, not just the script

The script path being absolute is not enough. A command's data arguments resolve
against the terminal's working directory, which is not necessarily the directory
a sibling `write_file` call just wrote to. Whenever a step writes a file with one
tool and reads it with another, give both an absolute path — or set the working
directory explicitly and keep every argument relative to that one directory.

## Working directories

If a command needs a scratch directory rather than a script path, declare it on
the terminal call instead of assuming one. Some runtimes expose a semantic
`agent_output` workdir that resolves to the agent's own writable output
directory; where it is unavailable, pass an absolute path. Do not write skills
that depend on an undeclared working directory.

## Where the variable does not reach

Writing it in either of these leaves the literal text on screen:

- **Frontmatter** (`setup.help` and friends). Substitution runs over the body
  that `skill_commands` loads; setup metadata is read straight off the parsed
  YAML. Describe the script by name there and keep the runnable command in the
  body.
- **Supporting files** under `references/`, `templates/` and package `README`s —
  including this one. `skill_view` reads them verbatim. Either keep their paths
  relative and say once, in SKILL.md, that they are relative to the package
  root, or repeat the runnable form in SKILL.md itself.

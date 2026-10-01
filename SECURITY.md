# Security policy

## Supported versions

misgrade has no release yet. Once it does, only the latest release on PyPI gets security fixes.

## Reporting a vulnerability

Please report it privately through the repository's
[Security tab](https://github.com/MohammadHijjawi97/misgrade/security) ("Report a
vulnerability"). If that option is not available, open an issue that asks for a private
contact, without any details of the problem.

Useful to include: the misgrade version, the command, Python API call or MCP tool call, and a
small grader or seed file that triggers the problem.

## Scope

**misgrade runs the grader you give it.** Auditing a grader means importing and calling its
code (in a separate worker process, which is isolation from misgrade's own state, not a
sandbox). Only audit graders you would run yourself.

Beyond that, misgrade is designed to:

- never execute the contents of seed files or of the responses it generates: they are data,
  passed to the grader as strings;
- make no network requests, except optional dataset loaders you call explicitly;
- make no model calls: the injection and "master key" cases are strings handed to the grader
  under test; if that grader calls a model, that is the grader's doing;
- write only the output files you ask for (by default under `misgrade-out/`; nothing with
  `--format none`, as the pre-commit hook runs).

The MCP server (`misgrade mcp`) speaks MCP over stdio only: it opens no network port. Its
`audit_grader` tool imports and runs the grader the client names, exactly as `misgrade audit`
would, so an agent connected to it can run any code your user account can import. Connect it
only to agents you would let run commands in that environment. Its other tools only read
misgrade's own data or a result file the client names.

The self-test (`misgrade selftest`) runs misgrade's own toy graders. Two of them model grader
processes that leave a stale lock behind when they are killed: they create
`misgrade-selftest-*.marker` files in the temporary directory and remove them again.

The GitHub Action passes its inputs to its scripts only through environment variables, so a
grader path or gate text cannot inject shell code, and pins every third-party action to a
commit SHA.

A way to make misgrade break one of these, or to make the MCP server run code other than the
grader the client named, is a vulnerability.

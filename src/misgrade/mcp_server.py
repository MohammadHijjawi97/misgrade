"""The MCP server: lets a coding agent audit the reward function or grader it just wrote.

Owner: builder D. Needs the optional ``mcp`` package (``pip install "misgrade[mcp]"``, mcp 2.x:
``mcp.server.mcpserver.MCPServer``); imported only by ``misgrade mcp``.

Planned tools (all read-only towards the user's files; they run the grader under test):

- ``audit_grader(target, adapter=None, answer_type=None, template=None, budget=300, seed=0)``:
  the Markdown summary plus the grader card, with the minimized findings and their
  certificates, so the agent can fix the grader and audit again.
- ``list_operators(answer_type=None)``: what misgrade will try for a type.
- ``explain_category(category)``: what a category means and the usual hardening fix.
"""

from __future__ import annotations

import sys
from typing import Any, Final

from misgrade.models import ExitCode

__all__ = ["INSTRUCTIONS", "SERVER_NAME", "build_server", "main"]

__stub__ = True

SERVER_NAME: Final = "misgrade"
INSTRUCTIONS: Final = (
    "misgrade audits a grader (a reward function, verifier or eval scorer) by grading answers "
    "certified equivalent to the gold (they must be accepted) and answers certified wrong "
    "(they must be rejected). Call audit_grader after writing or changing a grader; each "
    "finding comes with a minimized response and the certificate that says why the verdict "
    "is wrong."
)


def build_server() -> Any:
    """The MCP server with misgrade's tools registered. Raises ImportError when the ``mcp``
    package is missing."""
    raise NotImplementedError("builder D: mcp_server.build_server")


def main() -> int:
    """Run the server on stdio until the client disconnects."""
    try:
        server = build_server()
    except ImportError as exc:
        print(
            f'misgrade mcp needs the MCP SDK: pip install "misgrade[mcp]" ({exc})',
            file=sys.stderr,
        )
        return ExitCode.USAGE
    server.run()
    return ExitCode.OK

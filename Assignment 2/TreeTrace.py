"""Store and print the part of the search tree that was explored."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    kind: str
    depth: int
    edge: str = ""
    value: float = 0.0
    note: str = ""
    children: list[TreeNode] = field(default_factory=list)
    pruned_columns: list[int] = field(default_factory=list)


def format_tree(root: TreeNode) -> str:
    """Columns are zero-based. A cutoff value can be a bound, not an exact value."""
    lines = []

    def visit(node, prefix="", connector="", child_prefix=""):
        edge = f"{node.edge} | " if node.edge else ""
        note = f" ({node.note})" if node.note else ""
        lines.append(
            f"{prefix}{connector}{edge}{node.kind} depth={node.depth} "
            f"value={node.value:.6g}{note}"
        )
        branches = list(node.children)
        if node.pruned_columns:
            branches.append(None)

        for index, child in enumerate(branches):
            last = index == len(branches) - 1
            branch = "`- " if last else "+- "
            continuation = "   " if last else "|  "
            next_prefix = prefix + child_prefix
            if child is None:
                lines.append(
                    f"{next_prefix}{branch}PRUNED columns={node.pruned_columns}"
                )
            else:
                visit(child, next_prefix, branch, continuation)

    visit(root)
    return "\n".join(lines)

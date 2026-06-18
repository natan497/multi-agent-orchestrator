"""A safe arithmetic calculator tool.

Evaluates expressions via the ``ast`` module with an allowlist of node types — never
``eval`` — so untrusted model output can't execute arbitrary code.
"""

from __future__ import annotations

import ast
import operator
from typing import ClassVar

from pydantic import BaseModel, Field

from tools.base import Tool

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# Guard against trivially expensive exponents (e.g. 2**10**9).
_MAX_EXPONENT = 1000


class CalculatorArgs(BaseModel):
    expression: str = Field(
        description="A basic arithmetic expression, e.g. '2 * (3 + 4)' or '10 / 4'.",
    )


class Calculator(Tool):
    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Evaluate a basic arithmetic expression. Supports + - * / // % ** and parentheses. "
        "Use for any numeric computation instead of guessing."
    )
    Args: ClassVar[type[BaseModel]] = CalculatorArgs

    def run(self, args: CalculatorArgs) -> str:
        result = _safe_eval(args.expression)
        # Render integers without a trailing .0 for clean output.
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)


def _safe_eval(expression: str) -> float | int:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"could not parse expression: {expression!r}") from e
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if op is operator.pow and isinstance(right, (int, float)) and right > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator {type(node.op).__name__} is not allowed")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")

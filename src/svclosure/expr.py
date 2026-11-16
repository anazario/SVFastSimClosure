"""Validated translation of the LLPCombine C++ selection subset into columnar evaluation.

LLPCombine hands each bin's combined cut string to ``RDataFrame::Filter``, which
JIT-compiles it as C++.  Rather than ``eval``-ing a rewritten string, this module
parses the string with a small recursive-descent parser that accepts *only* the
grammar actually used by the SV bins of the analysis configs:

    or      := and ( '||' and )*
    and     := eq  ( '&&' eq )*
    eq      := rel ( ('=='|'!=') rel )?
    rel     := unary ( ('<'|'<='|'>'|'>=') unary )?
    unary   := '!' unary | '-' unary | primary
    primary := NUMBER | 'true' | 'false' | IDENT ( '[' INT ']' )? | '(' or ')'

Anything else (arithmetic, function calls, chained comparisons, ...) raises
:class:`UnsupportedExpression`; such bins must go through the RDataFrame backend.

C++ semantics that are reproduced deliberately:

* Operator precedence follows C++ (``!`` > relational > equality > ``&&`` > ``||``).
* Comparisons are done in float64.  A ``float`` branch compared with a ``double``
  literal is promoted to double in C++; NumPy 2 would instead cast the Python
  literal down to float32 (NEP 50), which flips events sitting exactly on a
  boundary such as ``rjr_Rs[0] >= 0.15``.
* A bare numeric operand used as a boolean is ``!= 0``.
* ``x[i]`` on an ``RVec`` is unchecked in C++ (undefined behaviour when out of
  range).  Here an out-of-range element becomes NaN, so it fails every
  comparison, and the number of such events is counted so it can be reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

import awkward as ak
import numpy as np

TRIGGER_RE = re.compile(r"\bTrigger_\w+\b")


class UnsupportedExpression(ValueError):
    """The expression uses syntax outside the validated subset."""


def strip_triggers(expression: str) -> str:
    """Replace every ``Trigger_*`` token by ``true``.

    Mirrors ``BuildFitInput::FilterRegions`` exactly (``std::regex
    "\\bTrigger_\\w+\\b"`` -> ``"true"``), which LLPCombine applies to signal
    samples whose ``kuSkimConfigTree.sMCType == 2`` (FastSim).
    """
    return TRIGGER_RE.sub("true", expression)


# ----------------------------------------------------------------------------- tokens

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<ident>[A-Za-z_]\w*)
  | (?P<op>&&|\|\||==|!=|<=|>=|<|>|!|-|\(|\)|\[|\])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise UnsupportedExpression(f"unexpected character {src[pos]!r} at {pos} in: {src}")
        kind = m.lastgroup
        if kind != "ws":
            tokens.append(Token(kind, m.group(), pos))
        pos = m.end()
    return tokens


# ----------------------------------------------------------------------------- AST


class Node:
    def cpp(self) -> str:  # canonical, fully parenthesised C++ rendering
        raise NotImplementedError

    def branches(self) -> set[str]:
        return set()

    def children(self) -> tuple["Node", ...]:
        return ()


@dataclass(frozen=True)
class Num(Node):
    value: float
    text: str

    def cpp(self) -> str:
        return self.text


@dataclass(frozen=True)
class Bool(Node):
    value: bool

    def cpp(self) -> str:
        return "true" if self.value else "false"


@dataclass(frozen=True)
class Var(Node):
    name: str
    index: int | None = None

    def cpp(self) -> str:
        return self.name if self.index is None else f"{self.name}[{self.index}]"

    def branches(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class Not(Node):
    operand: Node

    def cpp(self) -> str:
        return f"!({self.operand.cpp()})"

    def branches(self) -> set[str]:
        return self.operand.branches()

    def children(self):
        return (self.operand,)


@dataclass(frozen=True)
class Neg(Node):
    operand: Node

    def cpp(self) -> str:
        return f"-({self.operand.cpp()})"

    def branches(self) -> set[str]:
        return self.operand.branches()

    def children(self):
        return (self.operand,)


@dataclass(frozen=True)
class Cmp(Node):
    op: str
    left: Node
    right: Node

    def cpp(self) -> str:
        return f"({self.left.cpp()} {self.op} {self.right.cpp()})"

    def branches(self) -> set[str]:
        return self.left.branches() | self.right.branches()

    def children(self):
        return (self.left, self.right)


@dataclass(frozen=True)
class And(Node):
    terms: tuple[Node, ...]

    def cpp(self) -> str:
        return "(" + " && ".join(t.cpp() for t in self.terms) + ")"

    def branches(self) -> set[str]:
        return set().union(*(t.branches() for t in self.terms))

    def children(self):
        return self.terms


@dataclass(frozen=True)
class Or(Node):
    terms: tuple[Node, ...]

    def cpp(self) -> str:
        return "(" + " || ".join(t.cpp() for t in self.terms) + ")"

    def branches(self) -> set[str]:
        return set().union(*(t.branches() for t in self.terms))

    def children(self):
        return self.terms


# ----------------------------------------------------------------------------- parser


class _Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks = tokenize(src)
        self.i = 0

    def peek(self) -> Token | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, text: str | None = None, kind: str | None = None) -> Token:
        tok = self.peek()
        if tok is None or (text is not None and tok.text != text) or (kind is not None and tok.kind != kind):
            want = text or kind
            got = "end of input" if tok is None else repr(tok.text)
            where = len(self.src) if tok is None else tok.pos
            raise UnsupportedExpression(f"expected {want!r}, got {got} at {where} in: {self.src}")
        self.i += 1
        return tok

    def at(self, *texts: str) -> bool:
        tok = self.peek()
        return tok is not None and tok.kind == "op" and tok.text in texts

    def parse(self) -> Node:
        node = self.or_()
        if self.peek() is not None:
            tok = self.peek()
            raise UnsupportedExpression(f"unexpected {tok.text!r} at {tok.pos} in: {self.src}")
        return node

    def or_(self) -> Node:
        terms = [self.and_()]
        while self.at("||"):
            self.take("||")
            terms.append(self.and_())
        return terms[0] if len(terms) == 1 else Or(tuple(terms))

    def and_(self) -> Node:
        terms = [self.eq()]
        while self.at("&&"):
            self.take("&&")
            terms.append(self.eq())
        return terms[0] if len(terms) == 1 else And(tuple(terms))

    def eq(self) -> Node:
        left = self.rel()
        if self.at("==", "!="):
            op = self.take().text
            right = self.rel()
            if self.at("==", "!="):
                raise UnsupportedExpression(f"chained equality in: {self.src}")
            return Cmp(op, left, right)
        return left

    def rel(self) -> Node:
        left = self.unary()
        if self.at("<", "<=", ">", ">="):
            op = self.take().text
            right = self.unary()
            if self.at("<", "<=", ">", ">="):
                raise UnsupportedExpression(f"chained comparison in: {self.src}")
            return Cmp(op, left, right)
        return left

    def unary(self) -> Node:
        if self.at("!"):
            self.take("!")
            return Not(self.unary())
        if self.at("-"):
            self.take("-")
            inner = self.unary()
            if isinstance(inner, Num):
                return Num(-inner.value, "-" + inner.text)
            return Neg(inner)
        return self.primary()

    def primary(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise UnsupportedExpression(f"unexpected end of input in: {self.src}")
        if tok.kind == "num":
            self.take()
            return Num(float(tok.text), tok.text)
        if tok.kind == "ident":
            self.take()
            if tok.text in ("true", "false"):
                return Bool(tok.text == "true")
            if self.at("("):
                raise UnsupportedExpression(f"function call {tok.text}(...) is not supported in: {self.src}")
            if self.at("["):
                self.take("[")
                idx = self.take(kind="num")
                if not re.fullmatch(r"\d+", idx.text):
                    raise UnsupportedExpression(f"non-integer index {idx.text} in: {self.src}")
                self.take("]")
                return Var(tok.text, int(idx.text))
            return Var(tok.text)
        if tok.text == "(":
            self.take("(")
            node = self.or_()
            self.take(")")
            return node
        raise UnsupportedExpression(f"unexpected {tok.text!r} at {tok.pos} in: {self.src}")


def parse(src: str) -> Node:
    return _Parser(src).parse()


# ----------------------------------------------------------------------------- evaluation


@dataclass
class EvalContext:
    """Per-chunk evaluation state: branch arrays, a sub-expression cache, diagnostics.

    The cache is keyed on the canonical C++ rendering of each sub-tree, so the many
    identical baseline/cleaning terms shared by the ~40 bins are evaluated once per
    chunk.
    """

    arrays: Mapping[str, object]
    n_events: int
    cache: dict[str, np.ndarray] = field(default_factory=dict)  # numeric columns
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # boolean sub-expressions
    out_of_range: dict[str, int] = field(default_factory=dict)

    def column(self, var: Var) -> np.ndarray:
        key = var.cpp()
        if key in self.cache:
            return self.cache[key]
        raw = self.arrays[var.name]
        if var.index is None:
            col = np.asarray(ak.to_numpy(raw) if isinstance(raw, ak.Array) else raw)
            if col.ndim != 1:
                raise UnsupportedExpression(f"branch {var.name} is not scalar but used without an index")
        else:
            arr = raw if isinstance(raw, ak.Array) else ak.Array(raw)
            counts = ak.to_numpy(ak.num(arr, axis=1))
            missing = int(np.count_nonzero(counts <= var.index))
            if missing:
                self.out_of_range[key] = self.out_of_range.get(key, 0) + missing
            padded = ak.pad_none(arr, var.index + 1, axis=1, clip=True)[:, var.index]
            col = ak.to_numpy(ak.fill_none(ak.values_astype(padded, np.float64), np.nan))
        self.cache[key] = col
        return col


def _as_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64, copy=False)


def _as_bool(x: np.ndarray) -> np.ndarray:
    return x if x.dtype == np.bool_ else (x != 0)


_CMP: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "==": np.equal,
    "!=": np.not_equal,
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


def _value(node: Node, ctx: EvalContext) -> np.ndarray:
    """Numeric (or boolean) value of a node, as an array of length n_events."""
    if isinstance(node, Num):
        return np.full(ctx.n_events, node.value, dtype=np.float64)
    if isinstance(node, Bool):
        return np.full(ctx.n_events, node.value, dtype=np.bool_)
    if isinstance(node, Var):
        return ctx.column(node)
    if isinstance(node, Neg):
        return -_as_float(_value(node.operand, ctx))
    return evaluate(node, ctx)


def evaluate(node: Node, ctx: EvalContext) -> np.ndarray:
    """Boolean mask of ``node`` over the chunk in ``ctx``."""
    key = node.cpp()
    hit = ctx.masks.get(key)
    if hit is not None:
        return hit
    if isinstance(node, (Num, Var, Neg)):
        out = _as_bool(_value(node, ctx))
    elif isinstance(node, Bool):
        out = np.full(ctx.n_events, node.value, dtype=np.bool_)
    elif isinstance(node, Not):
        out = ~evaluate(node.operand, ctx)
    elif isinstance(node, Cmp):
        if isinstance(node.right, Num) and not isinstance(node.left, Num):
            left = _as_float(_value(node.left, ctx))
            out = _CMP[node.op](left, node.right.value)
        else:
            out = _CMP[node.op](_as_float(_value(node.left, ctx)), _as_float(_value(node.right, ctx)))
    elif isinstance(node, And):
        out = evaluate(node.terms[0], ctx).copy()
        for t in node.terms[1:]:
            out &= evaluate(t, ctx)
    elif isinstance(node, Or):
        out = evaluate(node.terms[0], ctx).copy()
        for t in node.terms[1:]:
            out |= evaluate(t, ctx)
    else:  # pragma: no cover
        raise UnsupportedExpression(f"unknown node {node!r}")
    ctx.masks[key] = out
    return out


# ----------------------------------------------------------------------------- analysis helpers


def conjuncts(node: Node) -> list[Node]:
    """Flatten the top-level ``&&`` structure."""
    if isinstance(node, And):
        out: list[Node] = []
        for t in node.terms:
            out.extend(conjuncts(t))
        return out
    return [node]


@dataclass
class Interval:
    lo: float = -np.inf
    lo_closed: bool = False
    hi: float = np.inf
    hi_closed: bool = False

    def apply(self, op: str, value: float) -> None:
        if op in (">", ">="):
            closed = op == ">="
            if value > self.lo or (value == self.lo and not closed):
                self.lo, self.lo_closed = value, closed
        elif op in ("<", "<="):
            closed = op == "<="
            if value < self.hi or (value == self.hi and not closed):
                self.hi, self.hi_closed = value, closed
        elif op == "==":
            self.lo = self.hi = value
            self.lo_closed = self.hi_closed = True

    @property
    def bounded(self) -> bool:
        return np.isfinite(self.lo) or np.isfinite(self.hi)


_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


def variable_interval(node: Node, variable: str) -> Interval:
    """Intersection of all top-level ``variable <op> constant`` requirements.

    Used only to *label* bins from their actual YAML selections, never to select.
    """
    iv = Interval()
    for term in conjuncts(node):
        if not isinstance(term, Cmp):
            continue
        if isinstance(term.left, Var) and term.left.cpp() == variable and isinstance(term.right, Num):
            iv.apply(term.op, term.right.value)
        elif isinstance(term.right, Var) and term.right.cpp() == variable and isinstance(term.left, Num):
            iv.apply(_FLIP[term.op], term.left.value)
    return iv

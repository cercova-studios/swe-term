"""L0 grammar + L1 lowering rules for Python. Everything below this
module (core.py) is language-agnostic and untouched by this file's
existence -- that is the entire claim Phase 4 exists to test.

Genuinely different from TypeScript's grammar, not a copy with renamed
node types:
  - Python has no separate "method definition" node -- a method is a
    function_definition lexically nested inside a class body. `kind` is
    derived from scope (are we inside a class right now?), not from node
    type alone, the way TypeScript's method_definition node lets us do it
    for free.
  - Decorators wrap the def in a decorated_definition node with no `name`
    field of its own -- the actual function_definition/class_definition is
    a child, and has to be found and descended into.
  - Call syntax differs at the node-type level: `call` (not
    call_expression) with an `attribute` node (not member_expression) for
    `obj.method()` calls, whose property field is named `attribute` (not
    `property`).
"""
from pathlib import Path

import tree_sitter_python as tspy
from tree_sitter import Language, Parser

LANGUAGE_NAME = "python"
FILE_GLOB = "*.py"
PY_LANGUAGE = Language(tspy.language())


def node_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def find_name(node, src: bytes):
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return node_text(src, name_node)
    return None


def extract_file(parser: Parser, root: Path, rel_path: str):
    src = (root / rel_path).read_bytes()
    tree = parser.parse(src)
    defs, calls = [], []

    def enclosing_symbol_id(stack):
        return stack[-1] if stack else f"{rel_path}::<module>"

    def walk(node, scope_stack, in_class):
        kind = node.type
        # decorated_definition has no def of its own -- its children are
        # the decorator expressions (which may contain calls, e.g. fastapi's
        # @app.get("/foo")) followed by the wrapped function/class_definition
        # child. Both are reached by the ordinary recursion below with no
        # special-casing needed; walk() just runs on every child either way.
        if kind in ("function_definition", "class_definition"):
            name = find_name(node, src) or "<anonymous>"
            symbol_id = f"{rel_path}:{node.start_point[0]+1}:{name}"
            def_kind = "class" if kind == "class_definition" else ("method" if in_class else "function")
            defs.append({
                "symbol_id": symbol_id, "file": rel_path,
                "kind": def_kind, "name": name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            })
            scope_stack = scope_stack + [symbol_id]
            in_class = (kind == "class_definition")
        elif kind == "call":
            callee = node.child_by_field_name("function")
            callee_name = None
            if callee is not None:
                if callee.type == "identifier":
                    callee_name = node_text(src, callee)
                elif callee.type == "attribute":
                    prop = callee.child_by_field_name("attribute")
                    if prop is not None:
                        callee_name = node_text(src, prop)
            if callee_name:
                calls.append({
                    "caller_symbol_id": enclosing_symbol_id(scope_stack),
                    "callee_name": callee_name, "file": rel_path,
                    "call_line": node.start_point[0] + 1,
                })
        for child in node.children:
            walk(child, scope_stack, in_class)

    walk(tree.root_node, [], False)
    return defs, calls


def make_parser():
    return Parser(PY_LANGUAGE)


def extract_many(root: Path, files):
    parser = make_parser()
    all_defs, all_calls = [], []
    for f in files:
        defs, calls = extract_file(parser, root, f)
        all_defs.extend(defs)
        all_calls.extend(calls)
    return all_defs, all_calls

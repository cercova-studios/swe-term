"""L0 grammar + L1 lowering rules for TypeScript. Everything below this
module (core.py) is language-agnostic; this is the ONLY file a TypeScript
integration should ever require."""
from pathlib import Path

import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

LANGUAGE_NAME = "typescript"
FILE_GLOB = "*.ts"
TS_LANGUAGE = Language(tsts.language_typescript())

DEF_KINDS = {
    "function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
}


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

    def walk(node, scope_stack):
        kind = node.type
        if kind in DEF_KINDS:
            name = find_name(node, src) or "<anonymous>"
            symbol_id = f"{rel_path}:{node.start_point[0]+1}:{name}"
            defs.append({
                "symbol_id": symbol_id, "file": rel_path,
                "kind": DEF_KINDS[kind], "name": name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            })
            scope_stack = scope_stack + [symbol_id]
        elif kind == "call_expression":
            callee = node.child_by_field_name("function")
            callee_name = None
            if callee is not None:
                if callee.type == "identifier":
                    callee_name = node_text(src, callee)
                elif callee.type == "member_expression":
                    prop = callee.child_by_field_name("property")
                    if prop is not None:
                        callee_name = node_text(src, prop)
            if callee_name:
                calls.append({
                    "caller_symbol_id": enclosing_symbol_id(scope_stack),
                    "callee_name": callee_name, "file": rel_path,
                    "call_line": node.start_point[0] + 1,
                })
        for child in node.children:
            walk(child, scope_stack)

    walk(tree.root_node, [])
    return defs, calls


def make_parser():
    return Parser(TS_LANGUAGE)


def extract_many(root: Path, files):
    parser = make_parser()
    all_defs, all_calls = [], []
    for f in files:
        defs, calls = extract_file(parser, root, f)
        all_defs.extend(defs)
        all_calls.extend(calls)
    return all_defs, all_calls

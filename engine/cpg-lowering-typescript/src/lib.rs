use cpg_schema::{Call, Def, ResolutionTier};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum LowerError {
    #[error("parser failed")]
    Parser,
}
pub fn lower(
    source: &str,
    file: impl Into<String>,
    tsx: bool,
) -> Result<(Vec<Def>, Vec<Call>), LowerError> {
    let mut parser = tree_sitter::Parser::new();
    let lang = if tsx {
        tree_sitter_typescript::LANGUAGE_TSX
    } else {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT
    };
    parser
        .set_language(&lang.into())
        .map_err(|_| LowerError::Parser)?;
    let tree = parser.parse(source, None).ok_or(LowerError::Parser)?;
    let file = file.into();
    let mut defs = Vec::new();
    let mut calls = Vec::new();
    walk(tree.root_node(), source, &file, None, &mut defs, &mut calls);
    Ok((defs, calls))
}
fn line(n: tree_sitter::Node) -> u32 {
    n.start_position().row as u32 + 1
}
fn walk(
    n: tree_sitter::Node,
    src: &str,
    file: &str,
    owner: Option<String>,
    defs: &mut Vec<Def>,
    calls: &mut Vec<Call>,
) {
    let kind = n.kind();
    let mut current = owner;
    if matches!(
        kind,
        "function_declaration"
            | "method_definition"
            | "class_declaration"
            | "interface_declaration"
            | "type_alias_declaration"
            | "enum_declaration"
            | "variable_declarator"
    ) {
        if let Some(name) = n.child_by_field_name("name") {
            let text = &src[name.byte_range()];
            defs.push(Def {
                name: text.to_string(),
                file: file.into(),
                line_start: line(n),
                line_end: n.end_position().row as u32 + 1,
                resolution_tier: ResolutionTier::SyntacticHeuristic,
            });
            current = Some(text.into());
        }
    }
    if kind == "call_expression" {
        if let Some(f) = n.child_by_field_name("function") {
            let callee_name = if f.kind() == "member_expression" {
                if let Some(prop) = f.child_by_field_name("property") {
                    &src[prop.byte_range()]
                } else {
                    &src[f.byte_range()]
                }
            } else {
                &src[f.byte_range()]
            };
            calls.push(Call {
                caller: current.clone().unwrap_or_else(|| "<module>".into()),
                callee: callee_name.into(),
                file: file.into(),
                line: line(n),
                resolution_tier: ResolutionTier::SyntacticHeuristic,
            });
        }
    }
    let mut c = n.walk();
    for ch in n.children(&mut c) {
        walk(ch, src, file, current.clone(), defs, calls);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lowers_defs_calls() {
        let (d, c) = lower(
            "function greet() { console.log('x'); foo(); }",
            "a.ts",
            false,
        )
        .unwrap();
        assert!(d.iter().any(|x| x.name == "greet"));
        assert!(c.iter().any(|x| x.callee == "foo"));
        assert!(c.iter().any(|x| x.callee == "log"));
    }
    #[test]
    fn parses_tsx() {
        let (d, c) = lower(
            "const App = () => { render(); return <div/>; };",
            "a.tsx",
            true,
        )
        .unwrap();
        assert!(d.iter().any(|x| x.name == "App"));
        assert!(c.iter().any(|x| x.caller == "App" && x.callee == "render"));
    }
    #[test]
    fn member_expression_callee() {
        let (_, c) = lower("obj.doThing()", "a.ts", false).unwrap();
        assert_eq!(c[0].callee, "doThing");
    }
}


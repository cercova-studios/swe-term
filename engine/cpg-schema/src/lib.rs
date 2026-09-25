use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum ResolutionTier {
    SyntacticHeuristic,
    Compiler,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Def {
    pub name: String,
    pub file: String,
    pub line_start: u32,
    pub line_end: u32,
    pub resolution_tier: ResolutionTier,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Call {
    pub caller: String,
    pub callee: String,
    pub file: String,
    pub line: u32,
    pub resolution_tier: ResolutionTier,
}
impl Def {
    pub fn key(&self) -> String {
        format!("{}:{}:{}", self.file, self.name, self.line_start)
    }
}
impl Call {
    pub fn key(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.file, self.caller, self.callee, self.line
        )
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpgSnapshot {
    pub schema_version: u32,
    pub revision: String,
    pub defs: Vec<Def>,
    pub calls: Vec<Call>,
}
impl CpgSnapshot {
    pub fn new(revision: impl Into<String>, defs: Vec<Def>, calls: Vec<Call>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            revision: revision.into(),
            defs,
            calls,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ids_are_stable() {
        let d = Def {
            name: "x".into(),
            file: "a.ts".into(),
            line_start: 2,
            line_end: 2,
            resolution_tier: ResolutionTier::SyntacticHeuristic,
        };
        assert_eq!(d.key(), "a.ts:x:2");
    }
    #[test]
    fn json_round_trip() {
        let s = CpgSnapshot::new("abc", vec![], vec![]);
        assert_eq!(
            serde_json::from_str::<CpgSnapshot>(&serde_json::to_string(&s).unwrap()).unwrap(),
            s
        );
    }
    #[test]
    fn tier_kebab_case() {
        assert_eq!(
            serde_json::to_string(&ResolutionTier::SyntacticHeuristic).unwrap(),
            "\"syntactic-heuristic\""
        );
        assert_eq!(
            serde_json::to_string(&ResolutionTier::Compiler).unwrap(),
            "\"compiler\""
        );
    }
}


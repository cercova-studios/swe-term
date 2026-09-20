use cpg_schema::CpgSnapshot;
use cpg_store::{Manifest, SnapshotStore, StoreError};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("stale base: expected {expected}, got {actual}")]
    Stale { expected: String, actual: String },
    #[error(transparent)]
    Store(#[from] StoreError),
}
pub struct Engine {
    store: SnapshotStore,
    layers: Vec<Manifest>,
    head: Option<String>,
}
impl Engine {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, EngineError> {
        Ok(Self {
            store: SnapshotStore::open(path)?,
            layers: Vec::new(),
            head: None,
        })
    }
    pub fn ingest_base(&mut self, s: CpgSnapshot) -> Result<(), EngineError> {
        let m = self.store.write_snapshot(&s)?;
        self.layers = vec![m];
        self.head = Some(s.revision);
        Ok(())
    }
    pub fn commit_overlay(&mut self, parent: &str, s: CpgSnapshot) -> Result<(), EngineError> {
        let actual = self.head.clone().unwrap_or_default();
        if actual != parent {
            return Err(EngineError::Stale {
                expected: actual,
                actual: parent.into(),
            });
        }
        let m = self.store.write_overlay(&s, parent)?;
        self.layers.push(m);
        self.head = Some(s.revision);
        Ok(())
    }
    pub fn differential_step(
        &mut self,
        parent: &str,
        revision: impl Into<String>,
        defs: Vec<cpg_schema::Def>,
        calls: Vec<cpg_schema::Call>,
    ) -> Result<(), EngineError> {
        self.commit_overlay(parent, CpgSnapshot::new(revision, defs, calls))
    }
    pub fn snapshot(&self) -> Result<CpgSnapshot, EngineError> {
        Ok(self.store.query(&self.layers)?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    #[test]
    fn rejects_stale() {
        let d = tempdir().unwrap();
        let mut e = Engine::open(d.path()).unwrap();
        e.ingest_base(CpgSnapshot::new("a", vec![], vec![]))
            .unwrap();
        assert!(matches!(
            e.commit_overlay("wrong", CpgSnapshot::new("b", vec![], vec![])),
            Err(EngineError::Stale { .. })
        ));
    }
}

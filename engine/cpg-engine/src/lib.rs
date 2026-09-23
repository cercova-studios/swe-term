use cpg_schema::CpgSnapshot;
use cpg_store::{Manifest, SnapshotStore, StoreError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("no base snapshot ingested")]
    NoBase,
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
        let store = SnapshotStore::open(path)?;
        let layers = store.discover_manifests()?;
        let head = layers.last().map(|m| m.revision.clone());
        Ok(Self {
            store,
            layers,
            head,
        })
    }

    pub fn head(&self) -> Option<&str> {
        self.head.as_deref()
    }

    pub fn ingest_base(&mut self, s: CpgSnapshot) -> Result<(), EngineError> {
        let m = self.store.write_snapshot(&s)?;
        self.layers = vec![m];
        self.head = Some(s.revision);
        Ok(())
    }

    pub fn commit_overlay(&mut self, parent: &str, s: CpgSnapshot) -> Result<(), EngineError> {
        let head = self.head.as_ref().ok_or(EngineError::NoBase)?;
        if head != parent {
            return Err(EngineError::Stale {
                expected: head.clone(),
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
        if self.layers.is_empty() {
            return Err(EngineError::NoBase);
        }
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

    #[test]
    fn rejects_overlay_without_base() {
        let d = tempdir().unwrap();
        let mut e = Engine::open(d.path()).unwrap();
        assert!(matches!(
            e.commit_overlay("", CpgSnapshot::new("b", vec![], vec![])),
            Err(EngineError::NoBase)
        ));
    }

    #[test]
    fn rejects_snapshot_without_base() {
        let d = tempdir().unwrap();
        let e = Engine::open(d.path()).unwrap();
        assert!(matches!(e.snapshot(), Err(EngineError::NoBase)));
    }

    #[test]
    fn reopens_populated_store() {
        let d = tempdir().unwrap();
        {
            let mut e = Engine::open(d.path()).unwrap();
            e.ingest_base(CpgSnapshot::new("rev1", vec![], vec![]))
                .unwrap();
            e.commit_overlay("rev1", CpgSnapshot::new("rev2", vec![], vec![]))
                .unwrap();
        }
        let e = Engine::open(d.path()).unwrap();
        assert_eq!(e.head(), Some("rev2"));
        let s = e.snapshot().unwrap();
        assert_eq!(s.revision, "rev2");
    }
}

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use cpg_schema::{Call, CpgSnapshot, Def, ResolutionTier, SCHEMA_VERSION};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("invalid schema version {0}")]
    Schema(u32),
    #[error("invalid revision: {0}")]
    InvalidRevision(String),
    #[error("invalid path: {0}")]
    InvalidPath(String),
    #[error("unknown resolution tier: {0}")]
    UnknownTier(String),
}

fn validate_revision(rev: &str) -> Result<(), StoreError> {
    if rev.is_empty()
        || rev.contains("..")
        || rev.contains('/')
        || rev.contains('\\')
        || rev.starts_with('.')
    {
        return Err(StoreError::InvalidRevision(rev.to_string()));
    }
    Ok(())
}

fn validate_filename(name: &str) -> Result<(), StoreError> {
    if name.is_empty()
        || name.contains("..")
        || name.contains('/')
        || name.contains('\\')
        || name.starts_with('.')
    {
        return Err(StoreError::InvalidPath(name.to_string()));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub schema_version: u32,
    pub revision: String,
    pub parent_revision: Option<String>,
    pub layer: u32,
    pub defs_file: String,
    pub calls_file: String,
}

#[derive(Clone, Debug)]
pub struct SnapshotStore {
    root: PathBuf,
}

impl SnapshotStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, StoreError> {
        fs::create_dir_all(root.as_ref())?;
        Ok(Self {
            root: root.as_ref().to_path_buf(),
        })
    }

    pub fn write_snapshot(&self, snapshot: &CpgSnapshot) -> Result<Manifest, StoreError> {
        self.write_layer(snapshot, None, 0)
    }

    pub fn write_overlay(
        &self,
        snapshot: &CpgSnapshot,
        parent: &str,
    ) -> Result<Manifest, StoreError> {
        self.write_layer(snapshot, Some(parent), 1)
    }

    fn write_layer(
        &self,
        s: &CpgSnapshot,
        parent: Option<&str>,
        layer: u32,
    ) -> Result<Manifest, StoreError> {
        if s.schema_version != SCHEMA_VERSION {
            return Err(StoreError::Schema(s.schema_version));
        }
        validate_revision(&s.revision)?;
        if let Some(p) = parent {
            validate_revision(p)?;
        }
        let stem = format!("{}-{}", s.revision, layer);
        let defs = format!("{stem}.defs.arrow");
        let calls = format!("{stem}.calls.arrow");
        write_defs(&self.root.join(&defs), &s.defs)?;
        write_calls(&self.root.join(&calls), &s.calls)?;
        let m = Manifest {
            schema_version: SCHEMA_VERSION,
            revision: s.revision.clone(),
            parent_revision: parent.map(str::to_owned),
            layer,
            defs_file: defs,
            calls_file: calls,
        };
        fs::write(
            self.root.join(format!("{stem}.manifest.json")),
            serde_json::to_vec_pretty(&m)?,
        )?;
        Ok(m)
    }

    pub fn discover_manifests(&self) -> Result<Vec<Manifest>, StoreError> {
        let mut manifests = Vec::new();
        if self.root.exists() {
            for entry in fs::read_dir(&self.root)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                        if name.ends_with(".manifest.json") {
                            let data = fs::read(&path)?;
                            let m: Manifest = serde_json::from_slice(&data)?;
                            manifests.push(m);
                        }
                    }
                }
            }
        }
        manifests.sort_by_key(|m| m.layer);
        Ok(manifests)
    }

    pub fn query(&self, manifests: &[Manifest]) -> Result<CpgSnapshot, StoreError> {
        let mut defs: HashMap<String, Def> = HashMap::new();
        let mut calls: HashMap<String, Call> = HashMap::new();
        for m in manifests {
            if m.schema_version != SCHEMA_VERSION {
                return Err(StoreError::Schema(m.schema_version));
            }
            validate_filename(&m.defs_file)?;
            validate_filename(&m.calls_file)?;
            for d in read_defs(&self.root.join(&m.defs_file))? {
                if d.name.is_empty() {
                    defs.retain(|_, old: &mut Def| {
                        !(old.file == d.file && (d.line_start == 0 || old.line_start == d.line_start))
                    });
                } else {
                    let k = d.key();
                    defs.insert(k, d);
                }
            }
            for c in read_calls(&self.root.join(&m.calls_file))? {
                if c.callee.is_empty() {
                    calls.retain(|_, old: &mut Call| {
                        !(old.file == c.file && (c.line == 0 || old.line == c.line))
                    });
                } else {
                    let k = c.key();
                    calls.insert(k, c);
                }
            }
        }
        let revision = manifests
            .last()
            .map(|m| m.revision.clone())
            .unwrap_or_default();
        let mut defs_vec: Vec<Def> = defs.into_values().collect();
        defs_vec.sort();
        let mut calls_vec: Vec<Call> = calls.into_values().collect();
        calls_vec.sort();
        Ok(CpgSnapshot::new(revision, defs_vec, calls_vec))
    }
}

fn schema(fields: Vec<Field>) -> Arc<Schema> {
    Arc::new(Schema::new(fields))
}

fn write_defs(path: &Path, rows: &[Def]) -> Result<(), StoreError> {
    let f = File::create(path)?;
    let sch = schema(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("file", DataType::Utf8, false),
        Field::new("start", DataType::UInt32, false),
        Field::new("end", DataType::UInt32, false),
        Field::new("tier", DataType::Utf8, false),
    ]);
    let b = RecordBatch::try_new(
        sch,
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|x| x.name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|x| x.file.as_str()),
            )),
            Arc::new(arrow::array::UInt32Array::from_iter_values(
                rows.iter().map(|x| x.line_start),
            )),
            Arc::new(arrow::array::UInt32Array::from_iter_values(
                rows.iter().map(|x| x.line_end),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|x| tier(x.resolution_tier.clone())),
            )),
        ],
    )?;
    let mut w = StreamWriter::try_new(f, &b.schema())?;
    w.write(&b)?;
    w.finish()?;
    Ok(())
}

fn write_calls(path: &Path, rows: &[Call]) -> Result<(), StoreError> {
    let f = File::create(path)?;
    let sch = schema(vec![
        Field::new("caller", DataType::Utf8, false),
        Field::new("callee", DataType::Utf8, false),
        Field::new("file", DataType::Utf8, false),
        Field::new("line", DataType::UInt32, false),
        Field::new("tier", DataType::Utf8, false),
    ]);
    let b = RecordBatch::try_new(
        sch,
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|x| x.caller.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|x| x.callee.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|x| x.file.as_str()),
            )),
            Arc::new(arrow::array::UInt32Array::from_iter_values(
                rows.iter().map(|x| x.line),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|x| tier(x.resolution_tier.clone())),
            )),
        ],
    )?;
    let mut w = StreamWriter::try_new(f, &b.schema())?;
    w.write(&b)?;
    w.finish()?;
    Ok(())
}

fn tier(t: ResolutionTier) -> &'static str {
    match t {
        ResolutionTier::SyntacticHeuristic => "syntactic-heuristic",
        ResolutionTier::Compiler => "compiler",
    }
}

fn read_defs(path: &Path) -> Result<Vec<Def>, StoreError> {
    let f = File::open(path)?;
    let m = unsafe { Mmap::map(&f)? };
    let mut r = StreamReader::try_new(Cursor::new(m), None)?;
    let mut out = Vec::new();
    while let Some(b) = r.next() {
        let b = b?;
        if b.num_columns() < 5 {
            return Err(StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                "defs batch missing required columns".into(),
            )));
        }
        let n = b
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 0 (name) is not StringArray".into(),
                ))
            })?;
        let fi = b
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 1 (file) is not StringArray".into(),
                ))
            })?;
        let st = b
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 2 (start) is not UInt32Array".into(),
                ))
            })?;
        let en = b
            .column(3)
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 3 (end) is not UInt32Array".into(),
                ))
            })?;
        let t = b
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 4 (tier) is not StringArray".into(),
                ))
            })?;
        for i in 0..b.num_rows() {
            out.push(Def {
                name: n.value(i).into(),
                file: fi.value(i).into(),
                line_start: st.value(i),
                line_end: en.value(i),
                resolution_tier: parse_tier(t.value(i))?,
            });
        }
    }
    Ok(out)
}

fn read_calls(path: &Path) -> Result<Vec<Call>, StoreError> {
    let f = File::open(path)?;
    let m = unsafe { Mmap::map(&f)? };
    let mut r = StreamReader::try_new(Cursor::new(m), None)?;
    let mut out = Vec::new();
    while let Some(b) = r.next() {
        let b = b?;
        if b.num_columns() < 5 {
            return Err(StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                "calls batch missing required columns".into(),
            )));
        }
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 0 (caller) is not StringArray".into(),
                ))
            })?;
        let c = b
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 1 (callee) is not StringArray".into(),
                ))
            })?;
        let fi = b
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 2 (file) is not StringArray".into(),
                ))
            })?;
        let l = b
            .column(3)
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 3 (line) is not UInt32Array".into(),
                ))
            })?;
        let t = b
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "column 4 (tier) is not StringArray".into(),
                ))
            })?;
        for i in 0..b.num_rows() {
            out.push(Call {
                caller: a.value(i).into(),
                callee: c.value(i).into(),
                file: fi.value(i).into(),
                line: l.value(i),
                resolution_tier: parse_tier(t.value(i))?,
            });
        }
    }
    Ok(out)
}

fn parse_tier(s: &str) -> Result<ResolutionTier, StoreError> {
    match s {
        "compiler" => Ok(ResolutionTier::Compiler),
        "syntactic-heuristic" => Ok(ResolutionTier::SyntacticHeuristic),
        other => Err(StoreError::UnknownTier(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn overlay_newest_wins() {
        let d = tempdir().unwrap();
        let st = SnapshotStore::open(d.path()).unwrap();
        let base = CpgSnapshot::new(
            "b",
            vec![Def {
                name: "x".into(),
                file: "a.ts".into(),
                line_start: 1,
                line_end: 1,
                resolution_tier: ResolutionTier::Compiler,
            }],
            vec![],
        );
        let over = CpgSnapshot::new(
            "h",
            vec![Def {
                name: "y".into(),
                file: "a.ts".into(),
                line_start: 2,
                line_end: 2,
                resolution_tier: ResolutionTier::Compiler,
            }],
            vec![],
        );
        let m1 = st.write_snapshot(&base).unwrap();
        let m2 = st.write_overlay(&over, "b").unwrap();
        let got = st.query(&[m1, m2]).unwrap();
        assert_eq!(got.defs.len(), 2);
        assert_eq!(got.revision, "h");
    }

    #[test]
    fn overlay_replaces_and_deletes() {
        let d = tempdir().unwrap();
        let st = SnapshotStore::open(d.path()).unwrap();
        let base = CpgSnapshot::new(
            "b",
            vec![
                Def {
                    name: "x".into(),
                    file: "a.ts".into(),
                    line_start: 1,
                    line_end: 1,
                    resolution_tier: ResolutionTier::Compiler,
                },
                Def {
                    name: "gone".into(),
                    file: "a.ts".into(),
                    line_start: 3,
                    line_end: 3,
                    resolution_tier: ResolutionTier::Compiler,
                },
            ],
            vec![],
        );
        let overlay = CpgSnapshot::new(
            "h",
            vec![
                Def {
                    name: "".into(),
                    file: "a.ts".into(),
                    line_start: 1,
                    line_end: 1,
                    resolution_tier: ResolutionTier::Compiler,
                },
                Def {
                    name: "x".into(),
                    file: "a.ts".into(),
                    line_start: 2,
                    line_end: 2,
                    resolution_tier: ResolutionTier::Compiler,
                },
                Def {
                    name: "".into(),
                    file: "a.ts".into(),
                    line_start: 3,
                    line_end: 3,
                    resolution_tier: ResolutionTier::Compiler,
                },
            ],
            vec![],
        );
        let m1 = st.write_snapshot(&base).unwrap();
        let m2 = st.write_overlay(&overlay, "b").unwrap();
        let got = st.query(&[m1, m2]).unwrap();
        assert_eq!(got.defs.len(), 1);
        assert_eq!(got.defs[0].line_start, 2);
    }

    #[test]
    fn rejects_invalid_revision() {
        let d = tempdir().unwrap();
        let st = SnapshotStore::open(d.path()).unwrap();
        let bad = CpgSnapshot::new("../escape", vec![], vec![]);
        assert!(matches!(
            st.write_snapshot(&bad),
            Err(StoreError::InvalidRevision(_))
        ));
    }

    #[test]
    fn deterministic_query_ordering() {
        let d = tempdir().unwrap();
        let st = SnapshotStore::open(d.path()).unwrap();
        let s = CpgSnapshot::new(
            "rev",
            vec![
                Def {
                    name: "z".into(),
                    file: "b.ts".into(),
                    line_start: 10,
                    line_end: 10,
                    resolution_tier: ResolutionTier::Compiler,
                },
                Def {
                    name: "a".into(),
                    file: "a.ts".into(),
                    line_start: 1,
                    line_end: 1,
                    resolution_tier: ResolutionTier::Compiler,
                },
            ],
            vec![],
        );
        let m = st.write_snapshot(&s).unwrap();
        let q1 = st.query(&[m.clone()]).unwrap();
        let q2 = st.query(&[m]).unwrap();
        assert_eq!(q1.defs, q2.defs);
        assert_eq!(q1.defs[0].name, "a");
        assert_eq!(q1.defs[1].name, "z");
    }
}

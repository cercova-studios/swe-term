# AST Service Architecture

## Overview

This document outlines the architecture for a cloud-based AST (Abstract Syntax Tree) parsing and code analysis service designed to support AI-powered PR review agents. The service runs on Google Cloud Run, processes repositories stored in Google Cloud Storage, and provides fast, scalable access to parsed code structure and dependency graphs.

## Table of Contents

1. [Core Principles](#core-principles)
2. [System Architecture](#system-architecture)
3. [Component Design](#component-design)
4. [Caching Strategy](#caching-strategy)
5. [Query Interface](#query-interface)
6. [Optimizations & Best Practices](#optimizations--best-practices)
7. [Monitoring & Operations](#monitoring--operations)
8. [Comparison with Aider](#comparison-with-aider)
9. [Future Enhancements](#future-enhancements)

---

## Core Principles

### 1. **Stateless & Scalable**
- Cloud Run instances can scale to zero or to hundreds based on demand
- No reliance on local disk persistence
- Shared state via Redis/Firestore

### 2. **Content-Addressable**
- Parse results keyed by content hash, not file path
- Reuse across branches, commits, and renamed files
- Efficient deduplication for vendored/copied code

### 3. **Pre-compute Heavy Operations**
- Async background parsing on repo ingestion
- Pre-computed graph metrics (PageRank, betweenness, communities)
- Lazy computation for cold repos

### 4. **Multi-Layer Caching**
- L1: In-process memory (microseconds)
- L2: Redis/Memorystore (milliseconds)
- L3: Parse from GCS (seconds)

### 5. **Incremental Updates**
- Only re-parse changed files
- Incremental graph updates for small changes
- Merkle trees for efficient change detection

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub / Git Provider                        │
└────────────────────────┬────────────────────────────────────────┘
                         │ Webhook (push/PR)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Cloud Pub/Sub / Tasks Queue                   │
│                    (Repo ingestion events)                       │
└────────────┬───────────────────────────────┬────────────────────┘
             │                               │
             │ Parse Job                     │ Update Job
             ▼                               ▼
┌────────────────────────┐     ┌────────────────────────────────┐
│  Background Parser     │     │  Incremental Update Worker     │
│  (Cloud Run Service)   │     │  (Cloud Run Service)           │
│                        │     │                                │
│  - Full repo scan      │     │  - Process git diffs           │
│  - Build initial graph │     │  - Update affected nodes       │
│  - Warm cache          │     │  - Recompute local metrics     │
└───────┬────────────────┘     └─────────┬──────────────────────┘
        │                                 │
        │ Read/Write                      │ Read/Write
        ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Google Cloud Storage                          │
│                    (Raw Repository Data)                         │
│  - /repos/{repo_id}/{commit_sha}/                               │
│  - Compressed archives or file trees                            │
└─────────────────────────────────────────────────────────────────┘
        │
        │ Read during parsing
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AST Parser Service                           │
│                     (Cloud Run - Main API)                       │
│                                                                  │
│  Endpoints:                                                      │
│  - POST /api/v1/ast/parse                                       │
│  - POST /api/v1/graph/dependencies                              │
│  - POST /api/v1/graph/important_files                           │
│  - POST /api/v1/repomap                                         │
│  - POST /api/v1/batch/*                                         │
└───────┬─────────────────────────────────────────┬───────────────┘
        │                                         │
        │ Cache Read/Write                        │ Graph Queries
        ▼                                         ▼
┌──────────────────────────┐     ┌───────────────────────────────┐
│  Cloud Memorystore       │     │  Neo4j / NetworkX             │
│  (Redis)                 │     │  (Graph Database - Optional)  │
│                          │     │                               │
│  Keys:                   │     │  Nodes: Files, Symbols        │
│  - ast:{hash} → tags     │     │  Edges: Dependencies, Refs    │
│  - graph:{repo}:{sha}    │     │  Metrics: Pre-computed        │
│  - symbols:{repo}:{sha}  │     │                               │
│  TTL: 7-30 days          │     │  For repos > 10k files        │
└──────────────────────────┘     └───────────────────────────────┘
        │
        │ Metrics & Logs
        ▼
┌─────────────────────────────────────────────────────────────────┐
│            Cloud Monitoring + Logging                            │
│  - Cache hit rates                                              │
│  - Parse latencies                                              │
│  - Graph computation times                                      │
│  - Error rates & types                                          │
└─────────────────────────────────────────────────────────────────┘
        │
        │ Consumed by
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PR Review Agent Service                         │
│                  (Cloud Run - Consumer)                          │
│                                                                  │
│  - Queries AST service for code context                         │
│  - Generates repo maps for LLM context                          │
│  - Analyzes PR changes with graph insights                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. Tree-Sitter Parser Module

**Purpose:** Parse source files into AST and extract tags (definitions/references)

```python
# parser.py
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from grep_ast import filename_to_lang
from grep_ast.tsl import USING_TSL_PACK, get_language, get_parser

warnings.simplefilter("ignore", category=FutureWarning)

@dataclass
class Tag:
    """Represents a code symbol (function, class, variable, etc.)"""
    rel_fname: str
    fname: str
    line: int
    name: str
    kind: str  # 'def' or 'ref'
    type: Optional[str] = None  # 'function', 'class', 'variable', etc.

class TreeSitterParser:
    """Parse source files using tree-sitter"""
    
    def __init__(self):
        self.query_cache = {}
        
    def get_scm_query(self, lang: str) -> Optional[str]:
        """Load tree-sitter query file for language"""
        if lang in self.query_cache:
            return self.query_cache[lang]
            
        if USING_TSL_PACK:
            subdir = "tree-sitter-language-pack"
        else:
            subdir = "tree-sitter-languages"
            
        query_path = Path(__file__).parent / "queries" / subdir / f"{lang}-tags.scm"
        
        if not query_path.exists():
            return None
            
        query_scm = query_path.read_text()
        self.query_cache[lang] = query_scm
        return query_scm
    
    def parse_file(
        self, 
        fname: str, 
        code: str, 
        rel_fname: Optional[str] = None
    ) -> List[Tag]:
        """Parse a single file and extract tags"""
        lang = filename_to_lang(fname)
        if not lang:
            return []
            
        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            print(f"Skipping file {fname}: {err}")
            return []
            
        query_scm = self.get_scm_query(lang)
        if not query_scm:
            return []
            
        # Parse code to AST
        tree = parser.parse(bytes(code, "utf-8"))
        
        # Run tag extraction queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)
        
        tags = []
        if USING_TSL_PACK:
            all_nodes = []
            for tag, nodes in captures.items():
                all_nodes += [(node, tag) for node in nodes]
        else:
            all_nodes = list(captures)
            
        for node, tag in all_nodes:
            if tag.startswith("name.definition."):
                kind = "def"
                symbol_type = tag.replace("name.definition.", "")
            elif tag.startswith("name.reference."):
                kind = "ref"
                symbol_type = tag.replace("name.reference.", "")
            else:
                continue
                
            tags.append(Tag(
                rel_fname=rel_fname or fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                type=symbol_type,
                line=node.start_point[0],
            ))
            
        return tags
```

### 2. Multi-Layer Cache

**Purpose:** Fast retrieval with minimal re-parsing

```python
# cache.py
import json
import hashlib
import asyncio
from typing import Optional, Dict, Any
from dataclasses import asdict
import redis.asyncio as aioredis

class ASTCache:
    """Multi-layer cache for parsed AST data"""
    
    def __init__(self, redis_url: str):
        self.memory = {}  # L1: In-process cache
        self.redis = aioredis.from_url(redis_url)  # L2: Shared cache
        self.parser = TreeSitterParser()
        
    @staticmethod
    def content_hash(content: str) -> str:
        """Generate content hash for cache key"""
        return hashlib.blake2b(content.encode('utf-8')).hexdigest()
    
    async def get_tags(
        self,
        repo_id: str,
        file_path: str,
        content: str
    ) -> List[Tag]:
        """Get parsed tags with multi-layer caching"""
        
        # Generate content-addressable key
        content_hash = self.content_hash(content)
        cache_key = f"ast:{repo_id}:{content_hash}"
        
        # L1: Check in-process memory
        if cache_key in self.memory:
            return self.memory[cache_key]
        
        # L2: Check Redis
        cached = await self.redis.get(cache_key)
        if cached:
            tags_data = json.loads(cached)
            tags = [Tag(**t) for t in tags_data]
            self.memory[cache_key] = tags  # Populate L1
            return tags
        
        # L3: Parse from source
        tags = self.parser.parse_file(file_path, content, file_path)
        
        # Store in caches
        tags_json = json.dumps([asdict(t) for t in tags])
        await self.redis.setex(cache_key, 7 * 24 * 3600, tags_json)  # 7 day TTL
        self.memory[cache_key] = tags
        
        return tags
    
    async def get_tags_batch(
        self,
        repo_id: str,
        files: Dict[str, str]  # {file_path: content}
    ) -> Dict[str, List[Tag]]:
        """Batch get tags for multiple files"""
        tasks = [
            self.get_tags(repo_id, path, content)
            for path, content in files.items()
        ]
        results = await asyncio.gather(*tasks)
        return dict(zip(files.keys(), results))
    
    async def invalidate(self, repo_id: str):
        """Invalidate all cache entries for a repo"""
        # Clear memory cache
        keys_to_delete = [k for k in self.memory.keys() if k.startswith(f"ast:{repo_id}:")]
        for key in keys_to_delete:
            del self.memory[key]
        
        # Clear Redis cache (use scan for large sets)
        async for key in self.redis.scan_iter(f"ast:{repo_id}:*"):
            await self.redis.delete(key)
```

### 3. Graph Builder & Ranker

**Purpose:** Build dependency graph and rank important symbols

```python
# graph.py
import math
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
import networkx as nx

class RepoGraph:
    """Build and analyze code dependency graphs"""
    
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
        
    async def build_graph(
        self,
        repo_id: str,
        commit_sha: str,
        all_tags: Dict[str, List[Tag]]  # {file_path: [tags]}
    ) -> nx.MultiDiGraph:
        """Build dependency graph from parsed tags"""
        
        # Check cache first
        cache_key = f"graph:{repo_id}:{commit_sha}"
        cached = await self.redis.get(cache_key)
        if cached:
            return pickle.loads(cached)
        
        # Build graph from tags
        G = nx.MultiDiGraph()
        defines = defaultdict(set)  # symbol -> {files that define it}
        references = defaultdict(list)  # symbol -> [files that reference it]
        definitions = defaultdict(set)  # (file, symbol) -> {Tag objects}
        
        # Process all tags
        for file_path, tags in all_tags.items():
            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(file_path)
                    definitions[(file_path, tag.name)].add(tag)
                elif tag.kind == "ref":
                    references[tag.name].append(file_path)
        
        # Add nodes
        for file_path in all_tags.keys():
            G.add_node(file_path, type='file')
        
        # Add edges based on references
        idents = set(defines.keys()).intersection(set(references.keys()))
        
        for ident in idents:
            definers = defines[ident]
            
            # Weight multiplier based on identifier characteristics
            mul = 1.0
            
            # Boost important identifiers
            is_snake = ("_" in ident) and any(c.isalpha() for c in ident)
            is_kebab = ("-" in ident) and any(c.isalpha() for c in ident)
            is_camel = any(c.isupper() for c in ident) and any(c.islower() for c in ident)
            
            if (is_snake or is_kebab or is_camel) and len(ident) >= 8:
                mul *= 10  # Long, well-named identifiers are important
            
            if ident.startswith("_"):
                mul *= 0.1  # Private identifiers less important
            
            if len(defines[ident]) > 5:
                mul *= 0.1  # Common symbols less distinctive
            
            # Create edges from referencers to definers
            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    weight = mul * math.sqrt(num_refs)
                    G.add_edge(
                        referencer, 
                        definer, 
                        weight=weight, 
                        ident=ident
                    )
        
        # Pre-compute graph metrics
        try:
            G.graph['pagerank'] = nx.pagerank(G, weight='weight')
            G.graph['betweenness'] = nx.betweenness_centrality(G, weight='weight')
            G.graph['communities'] = list(nx.community.louvain_communities(G))
        except:
            # Handle edge cases (empty graph, etc.)
            G.graph['pagerank'] = {}
            G.graph['betweenness'] = {}
            G.graph['communities'] = []
        
        # Cache the graph
        await self.redis.setex(
            cache_key,
            24 * 3600,  # 24 hour TTL
            pickle.dumps(G)
        )
        
        return G
    
    async def incremental_update(
        self,
        repo_id: str,
        commit_sha: str,
        changed_files: Dict[str, List[Tag]]  # {file_path: [new_tags]}
    ) -> nx.MultiDiGraph:
        """Update graph incrementally for changed files"""
        
        # Get existing graph
        cache_key = f"graph:{repo_id}:{commit_sha}"
        cached = await self.redis.get(cache_key)
        
        if not cached:
            # No existing graph, build from scratch
            return await self.build_graph(repo_id, commit_sha, changed_files)
        
        G = pickle.loads(cached)
        
        # Remove old edges for changed files
        for file_path in changed_files.keys():
            if file_path in G:
                # Store edges to potentially restore
                in_edges = list(G.in_edges(file_path, data=True))
                out_edges = list(G.out_edges(file_path, data=True))
                
                # Remove node and all edges
                G.remove_node(file_path)
        
        # Add updated files back
        # (Simplified - in production, would need full tag context)
        # This demonstrates the concept of incremental updates
        
        # Recompute only affected metrics
        # In practice, use approximate algorithms for large graphs
        G.graph['pagerank'] = nx.pagerank(G, weight='weight')
        
        # Update cache
        await self.redis.setex(cache_key, 24 * 3600, pickle.dumps(G))
        
        return G
    
    def rank_files(
        self,
        G: nx.MultiDiGraph,
        context_files: List[str] = None,
        strategy: str = 'pagerank'
    ) -> List[Tuple[str, float]]:
        """Rank files by importance"""
        
        if strategy == 'pagerank':
            if context_files:
                # Personalized PageRank
                personalization = {f: 1.0 for f in context_files if f in G}
                if personalization:
                    scores = nx.pagerank(G, personalization=personalization, weight='weight')
                else:
                    scores = G.graph.get('pagerank', {})
            else:
                scores = G.graph.get('pagerank', {})
        
        elif strategy == 'betweenness':
            scores = G.graph.get('betweenness', {})
        
        elif strategy == 'degree':
            scores = {node: G.degree(node, weight='weight') for node in G.nodes()}
        
        else:
            raise ValueError(f"Unknown ranking strategy: {strategy}")
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 4. Symbol Index

**Purpose:** Cross-repo symbol search

```python
# symbol_index.py
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class SymbolLocation:
    file: str
    line: int
    kind: str  # 'def' or 'ref'
    type: str  # 'function', 'class', etc.
    context: str = ""  # Surrounding code snippet

class SymbolIndex:
    """Inverted index for symbol lookup"""
    
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
    
    async def index_repo(
        self,
        repo_id: str,
        commit_sha: str,
        all_tags: Dict[str, List[Tag]],
        file_contents: Dict[str, str] = None
    ):
        """Build inverted index for all symbols"""
        
        index = defaultdict(list)
        
        for file_path, tags in all_tags.items():
            for tag in tags:
                # Get code context if available
                context = ""
                if file_contents and file_path in file_contents:
                    lines = file_contents[file_path].split('\n')
                    if 0 <= tag.line < len(lines):
                        # Get 3 lines of context
                        start = max(0, tag.line - 1)
                        end = min(len(lines), tag.line + 2)
                        context = '\n'.join(lines[start:end])
                
                location = {
                    'file': file_path,
                    'line': tag.line,
                    'kind': tag.kind,
                    'type': tag.type or 'unknown',
                    'context': context
                }
                index[tag.name].append(location)
        
        # Store in Redis
        cache_key = f"symbols:{repo_id}:{commit_sha}"
        await self.redis.setex(
            cache_key,
            24 * 3600,
            json.dumps({k: v for k, v in index.items()})
        )
    
    async def search(
        self,
        repo_id: str,
        commit_sha: str,
        symbol: str
    ) -> List[SymbolLocation]:
        """Find all locations of a symbol"""
        
        cache_key = f"symbols:{repo_id}:{commit_sha}"
        cached = await self.redis.get(cache_key)
        
        if not cached:
            return []
        
        index = json.loads(cached)
        locations_data = index.get(symbol, [])
        
        return [SymbolLocation(**loc) for loc in locations_data]
    
    async def fuzzy_search(
        self,
        repo_id: str,
        commit_sha: str,
        pattern: str,
        limit: int = 50
    ) -> Dict[str, List[SymbolLocation]]:
        """Fuzzy search for symbols matching pattern"""
        
        cache_key = f"symbols:{repo_id}:{commit_sha}"
        cached = await self.redis.get(cache_key)
        
        if not cached:
            return {}
        
        index = json.loads(cached)
        results = {}
        
        # Simple fuzzy matching (can be enhanced with more sophisticated algorithms)
        pattern_lower = pattern.lower()
        for symbol, locations in index.items():
            if pattern_lower in symbol.lower():
                results[symbol] = [SymbolLocation(**loc) for loc in locations]
                if len(results) >= limit:
                    break
        
        return results
```

### 5. API Service

**Purpose:** RESTful API for agent consumption

```python
# api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import time

app = FastAPI(title="AST Service API")

# Initialize components
ast_cache = ASTCache(redis_url=REDIS_URL)
repo_graph = RepoGraph(redis_url=REDIS_URL)
symbol_index = SymbolIndex(redis_url=REDIS_URL)

# Request/Response Models
class ParseRequest(BaseModel):
    repo_id: str
    file_path: str
    content: str

class ParseResponse(BaseModel):
    file_path: str
    tags: List[Dict]
    cache_hit: bool
    latency_ms: float

class DependenciesRequest(BaseModel):
    repo_id: str
    commit_sha: str
    file_path: str

class ImportantFilesRequest(BaseModel):
    repo_id: str
    commit_sha: str
    limit: int = 50
    context_files: Optional[List[str]] = None
    strategy: str = 'pagerank'

class RepoMapRequest(BaseModel):
    repo_id: str
    commit_sha: str
    context_files: List[str]
    max_tokens: int = 1024

# Endpoints
@app.post("/api/v1/ast/parse", response_model=ParseResponse)
async def parse_file(req: ParseRequest):
    """Parse a single file and return tags"""
    start = time.time()
    
    # Check cache before parsing
    content_hash = ASTCache.content_hash(req.content)
    cache_key = f"ast:{req.repo_id}:{content_hash}"
    cache_hit = cache_key in ast_cache.memory or await ast_cache.redis.exists(cache_key)
    
    tags = await ast_cache.get_tags(req.repo_id, req.file_path, req.content)
    
    return ParseResponse(
        file_path=req.file_path,
        tags=[asdict(t) for t in tags],
        cache_hit=cache_hit,
        latency_ms=(time.time() - start) * 1000
    )

@app.post("/api/v1/batch/parse")
async def batch_parse(repo_id: str, files: Dict[str, str]):
    """Parse multiple files in parallel"""
    results = await ast_cache.get_tags_batch(repo_id, files)
    return {
        path: [asdict(t) for t in tags]
        for path, tags in results.items()
    }

@app.post("/api/v1/graph/dependencies")
async def get_dependencies(req: DependenciesRequest):
    """Get files that this file depends on"""
    cache_key = f"graph:{req.repo_id}:{req.commit_sha}"
    cached = await repo_graph.redis.get(cache_key)
    
    if not cached:
        raise HTTPException(status_code=404, detail="Graph not found. Index repo first.")
    
    G = pickle.loads(cached)
    
    if req.file_path not in G:
        return []
    
    # Return outgoing edges (files this file depends on)
    return list(G.successors(req.file_path))

@app.post("/api/v1/graph/dependents")
async def get_dependents(req: DependenciesRequest):
    """Get files that depend on this file"""
    cache_key = f"graph:{req.repo_id}:{req.commit_sha}"
    cached = await repo_graph.redis.get(cache_key)
    
    if not cached:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    G = pickle.loads(cached)
    
    if req.file_path not in G:
        return []
    
    # Return incoming edges (files that depend on this file)
    return list(G.predecessors(req.file_path))

@app.post("/api/v1/graph/important_files")
async def get_important_files(req: ImportantFilesRequest):
    """Get ranked list of important files"""
    cache_key = f"graph:{req.repo_id}:{req.commit_sha}"
    cached = await repo_graph.redis.get(cache_key)
    
    if not cached:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    G = pickle.loads(cached)
    
    ranked = repo_graph.rank_files(
        G,
        context_files=req.context_files,
        strategy=req.strategy
    )
    
    return [
        {'file': file, 'score': score}
        for file, score in ranked[:req.limit]
    ]

@app.post("/api/v1/symbols/search")
async def search_symbol(repo_id: str, commit_sha: str, symbol: str):
    """Find all occurrences of a symbol"""
    locations = await symbol_index.search(repo_id, commit_sha, symbol)
    return [asdict(loc) for loc in locations]

@app.post("/api/v1/symbols/fuzzy")
async def fuzzy_search_symbols(
    repo_id: str,
    commit_sha: str,
    pattern: str,
    limit: int = 50
):
    """Fuzzy search for symbols"""
    results = await symbol_index.fuzzy_search(repo_id, commit_sha, pattern, limit)
    return {
        symbol: [asdict(loc) for loc in locations]
        for symbol, locations in results.items()
    }

@app.post("/api/v1/repomap")
async def generate_repomap(req: RepoMapRequest):
    """Generate Aider-style repository map"""
    # Implementation similar to Aider's get_ranked_tags_map
    # Combines graph ranking with tree formatting
    cache_key = f"graph:{req.repo_id}:{req.commit_sha}"
    cached = await repo_graph.redis.get(cache_key)
    
    if not cached:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    G = pickle.loads(cached)
    
    # Rank files
    ranked = repo_graph.rank_files(G, context_files=req.context_files)
    
    # Format as tree (simplified version)
    # In production, would use TreeContext like Aider
    output = []
    for file, score in ranked:
        output.append(f"\n{file}:")
        # Would add formatted symbols here
    
    repomap = '\n'.join(output)
    
    # Truncate to token limit (approximate)
    # In production, use actual tokenizer
    approx_tokens = len(repomap.split())
    if approx_tokens > req.max_tokens:
        # Binary search to fit token budget (like Aider)
        pass
    
    return {'repomap': repomap, 'tokens': approx_tokens}

# Background job endpoint
@app.post("/api/v1/jobs/index_repo")
async def index_repo(
    repo_id: str,
    commit_sha: str,
    background_tasks: BackgroundTasks
):
    """Trigger async repo indexing"""
    background_tasks.add_task(
        _index_repo_task,
        repo_id,
        commit_sha
    )
    return {'status': 'indexing', 'repo_id': repo_id, 'commit_sha': commit_sha}

async def _index_repo_task(repo_id: str, commit_sha: str):
    """Background task to index entire repo"""
    # 1. Download repo from GCS
    # 2. Parse all files
    # 3. Build graph
    # 4. Build symbol index
    # 5. Cache everything
    pass

# Metrics middleware
@app.middleware("http")
async def track_metrics(request, call_next):
    start = time.time()
    
    # Check if request hits cache
    cache_hit = False
    if hasattr(request.state, 'cache_hit'):
        cache_hit = request.state.cache_hit
    
    response = await call_next(request)
    
    latency_ms = (time.time() - start) * 1000
    
    # Log metrics (integrate with Cloud Monitoring)
    print(f"[METRIC] endpoint={request.url.path} latency={latency_ms:.2f}ms cache_hit={cache_hit}")
    
    return response
```

---

## Caching Strategy

### Content-Addressable Keys

```python
# Cache key structure
ast_key = f"ast:{repo_id}:{content_hash}"
graph_key = f"graph:{repo_id}:{commit_sha}"
symbols_key = f"symbols:{repo_id}:{commit_sha}"
```

### TTL Policy

| Cache Type | TTL | Rationale |
|------------|-----|-----------|
| AST tags | 7 days | Code doesn't change often, reusable across branches |
| Graphs | 1 day | Commit-specific, but commits are immutable |
| Symbols | 1 day | Same as graphs |
| Repo maps | 1 hour | Query-specific, regenerate fresh |

### Eviction Strategy

- **LRU eviction** in Redis when memory limit reached
- **Proactive cleanup** of old commits (>30 days)
- **Pin hot repos** with higher TTL or no expiration

### Cache Warming

```python
async def warm_cache_for_pr(repo_id: str, pr_number: int):
    """Pre-warm cache when PR is opened"""
    # 1. Get changed files from PR
    # 2. Parse all changed files
    # 3. Get dependencies of changed files
    # 4. Pre-compute personalized PageRank
    pass
```

---

## Query Interface

### Query Types & Use Cases

```python
# 1. Parse single file (on-demand)
POST /api/v1/ast/parse
{
  "repo_id": "myorg/myrepo",
  "file_path": "src/main.py",
  "content": "def main(): ..."
}

# 2. Get important files for PR context
POST /api/v1/graph/important_files
{
  "repo_id": "myorg/myrepo",
  "commit_sha": "abc123",
  "context_files": ["src/api/auth.py"],  # PR changed files
  "limit": 20,
  "strategy": "pagerank"
}

# 3. Find symbol definition
POST /api/v1/symbols/search
{
  "repo_id": "myorg/myrepo",
  "commit_sha": "abc123",
  "symbol": "authenticate_user"
}

# 4. Check dependencies
POST /api/v1/graph/dependencies
{
  "repo_id": "myorg/myrepo",
  "commit_sha": "abc123",
  "file_path": "src/api/users.py"
}

# 5. Generate repo map for LLM
POST /api/v1/repomap
{
  "repo_id": "myorg/myrepo",
  "commit_sha": "abc123",
  "context_files": ["src/api/auth.py"],
  "max_tokens": 2048
}

# 6. Batch operations
POST /api/v1/batch/parse
{
  "repo_id": "myorg/myrepo",
  "files": {
    "src/file1.py": "...",
    "src/file2.py": "..."
  }
}
```

---

## Optimizations & Best Practices

### 1. **Merkle Trees for Change Detection**

Based on research showing Cursor and other tools use Merkle trees:

```python
# merkle.py
import hashlib
from typing import Dict, List, Tuple

class MerkleTree:
    """Efficient change detection using Merkle trees"""
    
    @staticmethod
    def hash_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
    
    @staticmethod
    def hash_pair(left: str, right: str) -> str:
        return hashlib.sha256((left + right).encode()).hexdigest()
    
    def build_tree(self, files: Dict[str, str]) -> Dict[str, str]:
        """Build Merkle tree for repo state
        
        Returns:
            Dict mapping file paths to their Merkle hashes
        """
        # Sort files for consistent ordering
        sorted_files = sorted(files.items())
        
        # Leaf nodes: hash individual files
        leaves = {
            path: self.hash_content(content)
            for path, content in sorted_files
        }
        
        # Build tree bottom-up (simplified)
        # In production, would build proper tree structure
        all_hashes = ''.join(leaves.values())
        root_hash = hashlib.sha256(all_hashes.encode()).hexdigest()
        
        return {
            'root': root_hash,
            'leaves': leaves
        }
    
    def detect_changes(
        self,
        old_tree: Dict[str, str],
        new_files: Dict[str, str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Detect changed, added, and deleted files
        
        Returns:
            (changed, added, deleted) file paths
        """
        new_tree = self.build_tree(new_files)
        
        old_leaves = old_tree.get('leaves', {})
        new_leaves = new_tree['leaves']
        
        changed = []
        added = []
        deleted = []
        
        # Find changed and added
        for path, new_hash in new_leaves.items():
            if path not in old_leaves:
                added.append(path)
            elif old_leaves[path] != new_hash:
                changed.append(path)
        
        # Find deleted
        for path in old_leaves:
            if path not in new_leaves:
                deleted.append(path)
        
        return changed, added, deleted

# Usage in incremental updates
async def update_repo_incrementally(
    repo_id: str,
    old_commit: str,
    new_commit: str
):
    """Use Merkle tree to detect and update only changed files"""
    
    # Load old Merkle tree from cache
    old_tree_key = f"merkle:{repo_id}:{old_commit}"
    old_tree = await redis.get(old_tree_key)
    
    # Load new repo state
    new_files = await load_files_from_gcs(repo_id, new_commit)
    
    # Detect changes
    merkle = MerkleTree()
    changed, added, deleted = merkle.detect_changes(
        json.loads(old_tree) if old_tree else {},
        new_files
    )
    
    print(f"Changes: {len(changed)} modified, {len(added)} added, {len(deleted)} deleted")
    
    # Only process changed files
    files_to_process = {
        path: new_files[path]
        for path in changed + added
    }
    
    # Parse only changed files
    results = await ast_cache.get_tags_batch(repo_id, files_to_process)
    
    # Incrementally update graph
    await repo_graph.incremental_update(repo_id, new_commit, results)
    
    # Store new Merkle tree
    new_tree = merkle.build_tree(new_files)
    await redis.setex(
        f"merkle:{repo_id}:{new_commit}",
        30 * 24 * 3600,  # 30 days
        json.dumps(new_tree)
    )
```

### 2. **Hybrid Retrieval: Symbol + Vector Search**

Research shows vector search is 40% more efficient than grep-only:

```python
# hybrid_search.py
from typing import List, Tuple
import numpy as np

class HybridCodeSearch:
    """Combine symbol search with semantic vector search"""
    
    def __init__(self, symbol_index: SymbolIndex, embedding_service_url: str):
        self.symbol_index = symbol_index
        self.embedding_url = embedding_service_url
    
    async def search(
        self,
        repo_id: str,
        commit_sha: str,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float, str]]:
        """Hybrid search combining exact and semantic matching
        
        Returns:
            List of (file_path, score, match_type) tuples
        """
        results = []
        
        # 1. Exact symbol search (fast path)
        symbol_matches = await self.symbol_index.search(
            repo_id, commit_sha, query
        )
        
        for match in symbol_matches[:top_k]:
            results.append((match.file, 1.0, 'exact'))
        
        # 2. Fuzzy symbol search
        fuzzy_matches = await self.symbol_index.fuzzy_search(
            repo_id, commit_sha, query, limit=top_k
        )
        
        for symbol, locations in list(fuzzy_matches.items())[:5]:
            for loc in locations[:2]:  # Top 2 per symbol
                results.append((loc.file, 0.8, 'fuzzy'))
        
        # 3. Semantic vector search (for natural language queries)
        if len(query.split()) > 1:  # Multi-word query
            # Get embedding for query
            query_embedding = await self._get_embedding(query)
            
            # Search vector index
            vector_matches = await self._vector_search(
                repo_id, commit_sha, query_embedding, top_k=top_k
            )
            
            for file, score in vector_matches:
                results.append((file, score * 0.7, 'semantic'))
        
        # Deduplicate and rank
        seen = set()
        ranked = []
        for file, score, match_type in sorted(results, key=lambda x: -x[1]):
            if file not in seen:
                ranked.append((file, score, match_type))
                seen.add(file)
            if len(ranked) >= top_k:
                break
        
        return ranked
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from external service"""
        # In production, use proper embedding service
        # Options: OpenAI, Voyage, or local models via Ollama
        pass
    
    async def _vector_search(
        self,
        repo_id: str,
        commit_sha: str,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Search vector index for semantically similar code"""
        # Integration with vector DB (Milvus, Pinecone, etc.)
        pass
```

### 3. **Distributed Indexing**

For large repos, distribute parsing across multiple workers:

```python
# distributed_indexing.py
from google.cloud import tasks_v2
from typing import List

class DistributedIndexer:
    """Distribute parsing jobs across Cloud Run instances"""
    
    def __init__(self, project_id: str, queue_name: str):
        self.client = tasks_v2.CloudTasksClient()
        self.queue_path = self.client.queue_path(
            project_id, 'us-central1', queue_name
        )
    
    async def index_repo_distributed(
        self,
        repo_id: str,
        commit_sha: str,
        file_list: List[str],
        batch_size: int = 100
    ):
        """Split repo into batches and process in parallel"""
        
        # Divide files into batches
        batches = [
            file_list[i:i+batch_size]
            for i in range(0, len(file_list), batch_size)
        ]
        
        print(f"Created {len(batches)} batches of ~{batch_size} files each")
        
        # Create Cloud Task for each batch
        for i, batch in enumerate(batches):
            task = {
                'http_request': {
                    'http_method': tasks_v2.HttpMethod.POST,
                    'url': f'{SERVICE_URL}/internal/parse_batch',
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({
                        'repo_id': repo_id,
                        'commit_sha': commit_sha,
                        'files': batch,
                        'batch_id': i
                    }).encode()
                }
            }
            
            self.client.create_task(
                request={'parent': self.queue_path, 'task': task}
            )
        
        # Wait for all batches to complete
        # (Track progress in Redis)
        await self._wait_for_completion(repo_id, commit_sha, len(batches))
    
    async def _wait_for_completion(
        self,
        repo_id: str,
        commit_sha: str,
        total_batches: int
    ):
        """Wait for all batches to finish"""
        progress_key = f"index_progress:{repo_id}:{commit_sha}"
        
        while True:
            completed = await redis.get(progress_key)
            if completed and int(completed) >= total_batches:
                break
            await asyncio.sleep(1)
        
        # All batches done, build graph from results
        await self._finalize_index(repo_id, commit_sha)
```

### 4. **Query Performance Optimization**

```python
# query_optimizer.py

class QueryOptimizer:
    """Optimize query patterns for common use cases"""
    
    @staticmethod
    async def prefetch_for_pr_review(
        repo_id: str,
        commit_sha: str,
        pr_files: List[str]
    ):
        """Prefetch all data needed for PR review"""
        
        # Launch parallel queries
        tasks = [
            # 1. Parse all PR files
            ast_cache.get_tags_batch(repo_id, {f: "..." for f in pr_files}),
            
            # 2. Get dependency graph
            repo_graph.redis.get(f"graph:{repo_id}:{commit_sha}"),
            
            # 3. Get important files (personalized to PR)
            repo_graph.rank_files(G, context_files=pr_files),
            
            # 4. Prefetch common symbols
            _prefetch_common_symbols(repo_id, commit_sha, pr_files)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            'pr_tags': results[0],
            'graph': results[1],
            'important_files': results[2],
            'symbols': results[3]
        }
    
    @staticmethod
    async def _prefetch_common_symbols(
        repo_id: str,
        commit_sha: str,
        files: List[str]
    ):
        """Prefetch symbols likely to be queried"""
        # Get all definitions from PR files
        # Common patterns: class names, public functions
        pass
```

### 5. **Graph Database for Scale**

For repos > 10k files, use Neo4j instead of NetworkX:

```python
# neo4j_graph.py
from neo4j import AsyncGraphDatabase

class Neo4jRepoGraph:
    """Graph database for large-scale code graphs"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    async def build_graph(
        self,
        repo_id: str,
        commit_sha: str,
        all_tags: Dict[str, List[Tag]]
    ):
        """Build graph in Neo4j"""
        
        async with self.driver.session() as session:
            # Create file nodes
            await session.run("""
                UNWIND $files AS file
                MERGE (f:File {
                    path: file.path,
                    repo: $repo_id,
                    commit: $commit_sha
                })
            """, files=[{'path': p} for p in all_tags.keys()],
                repo_id=repo_id, commit_sha=commit_sha)
            
            # Create symbol nodes
            symbols = []
            for file_path, tags in all_tags.items():
                for tag in tags:
                    if tag.kind == 'def':
                        symbols.append({
                            'name': tag.name,
                            'type': tag.type,
                            'file': file_path,
                            'line': tag.line
                        })
            
            await session.run("""
                UNWIND $symbols AS sym
                MATCH (f:File {path: sym.file, repo: $repo_id})
                MERGE (s:Symbol {
                    name: sym.name,
                    repo: $repo_id,
                    commit: $commit_sha
                })
                ON CREATE SET s.type = sym.type
                MERGE (f)-[:DEFINES {line: sym.line}]->(s)
            """, symbols=symbols, repo_id=repo_id, commit_sha=commit_sha)
            
            # Create reference edges
            references = []
            for file_path, tags in all_tags.items():
                for tag in tags:
                    if tag.kind == 'ref':
                        references.append({
                            'symbol': tag.name,
                            'file': file_path,
                            'line': tag.line
                        })
            
            await session.run("""
                UNWIND $refs AS ref
                MATCH (f:File {path: ref.file, repo: $repo_id})
                MATCH (s:Symbol {name: ref.symbol, repo: $repo_id})
                MERGE (f)-[:REFERENCES {line: ref.line}]->(s)
            """, refs=references, repo_id=repo_id, commit_sha=commit_sha)
            
            # Run PageRank algorithm in Neo4j
            await session.run("""
                CALL gds.pageRank.write({
                    nodeProjection: 'File',
                    relationshipProjection: {
                        DEPENDS_ON: {
                            type: 'REFERENCES',
                            orientation: 'NATURAL',
                            properties: 'weight'
                        }
                    },
                    writeProperty: 'pagerank'
                })
            """)
    
    async def query_dependencies(
        self,
        repo_id: str,
        commit_sha: str,
        file_path: str,
        max_depth: int = 2
    ) -> List[str]:
        """Get transitive dependencies up to max_depth"""
        
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH path = (f:File {path: $file_path, repo: $repo_id})
                            -[:REFERENCES*1..$max_depth]->
                             (s:Symbol)<-[:DEFINES]-(dep:File)
                WHERE f <> dep
                RETURN DISTINCT dep.path AS dependency
            """, file_path=file_path, repo_id=repo_id, max_depth=max_depth)
            
            return [record['dependency'] async for record in result]
    
    async def custom_query(self, cypher: str, params: Dict):
        """Run custom Cypher queries for agent"""
        async with self.driver.session() as session:
            result = await session.run(cypher, **params)
            return [record.data() async for record in result]
```

---

## Monitoring & Operations

### Key Metrics to Track

```python
# metrics.py
from google.cloud import monitoring_v3
import time

class MetricsCollector:
    """Collect and report metrics to Cloud Monitoring"""
    
    def __init__(self, project_id: str):
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"
    
    def record_cache_hit(self, cache_layer: str, hit: bool):
        """Track cache hit rates"""
        # L1 (memory), L2 (redis), L3 (parse)
        metric = 'ast_service/cache/hits'
        self._write_metric(metric, 1 if hit else 0, {
            'layer': cache_layer,
            'result': 'hit' if hit else 'miss'
        })
    
    def record_parse_latency(self, language: str, latency_ms: float):
        """Track parsing performance by language"""
        metric = 'ast_service/parse/latency'
        self._write_metric(metric, latency_ms, {'language': language})
    
    def record_graph_size(self, repo_id: str, num_nodes: int, num_edges: int):
        """Track graph complexity"""
        self._write_metric('ast_service/graph/nodes', num_nodes, {'repo': repo_id})
        self._write_metric('ast_service/graph/edges', num_edges, {'repo': repo_id})
    
    def record_query_type(self, query_type: str, latency_ms: float):
        """Track API query patterns"""
        self._write_metric('ast_service/query/count', 1, {'type': query_type})
        self._write_metric('ast_service/query/latency', latency_ms, {'type': query_type})

# Dashboard queries
DASHBOARD_QUERIES = """
-- Cache hit rate by layer
SELECT
  ROUND(SUM(IF(result='hit', value, 0)) / SUM(value) * 100, 2) AS hit_rate_pct,
  layer
FROM metrics
WHERE metric = 'ast_service/cache/hits'
  AND timestamp > NOW() - INTERVAL 1 HOUR
GROUP BY layer;

-- Slowest languages to parse
SELECT
  language,
  AVG(value) AS avg_latency_ms,
  MAX(value) AS max_latency_ms,
  COUNT(*) AS parse_count
FROM metrics
WHERE metric = 'ast_service/parse/latency'
  AND timestamp > NOW() - INTERVAL 1 HOUR
GROUP BY language
ORDER BY avg_latency_ms DESC
LIMIT 10;

-- Most queried repos
SELECT
  repo,
  COUNT(*) AS query_count
FROM metrics
WHERE metric = 'ast_service/query/count'
  AND timestamp > NOW() - INTERVAL 1 DAY
GROUP BY repo
ORDER BY query_count DESC
LIMIT 20;
"""
```

### Alerting Rules

```yaml
# alerting.yaml
alerts:
  - name: high_cache_miss_rate
    condition: cache_hit_rate < 70%
    duration: 5m
    severity: warning
    description: "Cache hit rate dropped below 70%"
    
  - name: slow_parse_times
    condition: parse_latency_p95 > 5000ms
    duration: 5m
    severity: warning
    description: "95th percentile parse time exceeds 5 seconds"
    
  - name: graph_computation_timeout
    condition: graph_build_time > 30s
    duration: 1m
    severity: critical
    description: "Graph building taking too long"
    
  - name: redis_connection_errors
    condition: redis_errors > 10
    duration: 1m
    severity: critical
    description: "Multiple Redis connection failures"
```

### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    
    checks = {
        'status': 'healthy',
        'checks': {}
    }
    
    # Check Redis connectivity
    try:
        await redis.ping()
        checks['checks']['redis'] = 'ok'
    except Exception as e:
        checks['checks']['redis'] = f'error: {e}'
        checks['status'] = 'unhealthy'
    
    # Check GCS access
    try:
        # Test bucket access
        checks['checks']['gcs'] = 'ok'
    except Exception as e:
        checks['checks']['gcs'] = f'error: {e}'
        checks['status'] = 'unhealthy'
    
    # Check parser availability
    try:
        parser = TreeSitterParser()
        test_tags = parser.parse_file('test.py', 'def test(): pass', 'test.py')
        checks['checks']['parser'] = 'ok'
    except Exception as e:
        checks['checks']['parser'] = f'error: {e}'
        checks['status'] = 'degraded'
    
    # Report metrics
    checks['metrics'] = {
        'cache_size': len(ast_cache.memory),
        'uptime': time.time() - START_TIME
    }
    
    return checks
```

---

## Comparison with Aider

| Aspect | Aider | AST Service |
|--------|-------|-------------|
| **Deployment** | Local CLI tool | Cloud Run (stateless containers) |
| **Storage** | SQLite on disk (`.aider.tags.cache`) | Redis (shared) + GCS (repos) |
| **Concurrency** | Single user, sequential | Multi-tenant, parallel requests |
| **Scale** | One repo at a time | 100s of repos concurrently |
| **Cache Strategy** | Mtime-based invalidation | Content-hash + Merkle trees |
| **Graph Persistence** | In-memory NetworkX, ephemeral | Redis/Neo4j, persistent |
| **Query Interface** | Python API | REST API with batch support |
| **Latency** | Microseconds (local disk) | Milliseconds (network) |
| **Parsing Trigger** | On-demand per CLI invocation | Pre-compute + on-demand |
| **Graph Updates** | Full rebuild each time | Incremental updates |
| **Symbol Search** | File-scoped tags | Cross-repo symbol index |
| **Vector Search** | None | Optional hybrid search |
| **Use Case** | Interactive coding with AI | Automated PR reviews, agents |

---

## Future Enhancements

### 1. **Semantic Code Search**

Integrate embeddings for natural language queries:

```python
# Vector search for "authentication middleware"
results = await hybrid_search.search(
    repo_id="myorg/myrepo",
    commit_sha="abc123",
    query="authentication middleware that validates JWT tokens",
    top_k=10
)
# Returns semantically relevant code, not just keyword matches
```

### 2. **Cross-Repository Analysis**

Build global symbol index across all repos:

```python
# Find all implementations of "UserService" interface across org
results = await global_symbol_index.search(
    org_id="myorg",
    symbol="UserService",
    kind="implementation"
)
```

### 3. **Temporal Code Analysis**

Track how code evolves over time:

```python
# Show how function signature changed across commits
history = await get_symbol_history(
    repo_id="myorg/myrepo",
    symbol="authenticate_user",
    from_commit="v1.0.0",
    to_commit="HEAD"
)
```

### 4. **ML-Enhanced Ranking**

Train models on historical PR review data:

```python
# Predict which files are most likely to need review
predictions = await ml_ranker.predict_review_priority(
    repo_id="myorg/myrepo",
    pr_files=["src/auth.py", "src/users.py"],
    pr_description="Add OAuth support"
)
```

### 5. **Code Clone Detection**

Use AST similarity to find duplicated code:

```python
# Find similar code blocks across repo
clones = await clone_detector.find_clones(
    repo_id="myorg/myrepo",
    min_similarity=0.85,
    min_lines=10
)
```

### 6. **GraphQL Query Interface**

Flexible querying for agents:

```graphql
query PRContext {
  repo(id: "myorg/myrepo", commit: "abc123") {
    files(paths: ["src/auth.py"]) {
      path
      tags {
        name
        kind
        type
      }
      dependencies {
        path
        importance
      }
      dependents {
        path
      }
    }
    importantFiles(limit: 20, contextFiles: ["src/auth.py"]) {
      path
      score
    }
  }
}
```

---

## Deployment Checklist

### Infrastructure Setup

- [ ] Create Cloud Run services (parser, background worker)
- [ ] Provision Cloud Memorystore (Redis) instance
- [ ] Set up GCS bucket for repos
- [ ] Configure Cloud Tasks queue for async jobs
- [ ] Set up Cloud Monitoring dashboards
- [ ] Configure alerting rules
- [ ] Set up Cloud Logging sinks

### Security

- [ ] Enable VPC connector for Redis access
- [ ] Configure IAM roles (service accounts)
- [ ] Enable audit logging
- [ ] Set up secret management for API keys
- [ ] Configure CORS policies
- [ ] Enable rate limiting

### Performance

- [ ] Set min-instances=1 for low latency (optional)
- [ ] Configure concurrency limits (80-100 per instance)
- [ ] Set memory limits (2-4GB recommended)
- [ ] Configure Redis maxmemory-policy (allkeys-lru)
- [ ] Enable connection pooling

### Monitoring

- [ ] Set up SLIs/SLOs (latency, availability)
- [ ] Create uptime checks
- [ ] Configure error reporting
- [ ] Set up cost monitoring
- [ ] Create performance dashboards

---

## Conclusion

This architecture provides a scalable, efficient AST parsing and code analysis service optimized for AI-powered PR review agents. Key advantages:

1. **Fast**: Multi-layer caching, content-addressable storage, pre-computation
2. **Scalable**: Stateless Cloud Run, distributed indexing, Redis/Neo4j
3. **Efficient**: Merkle trees for change detection, incremental updates
4. **Flexible**: REST API, batch operations, hybrid search
5. **Observable**: Comprehensive metrics, health checks, alerting

The design incorporates best practices from Aider's proven approach while adapting for cloud-native, multi-tenant operation at scale.

---

## References

- [Aider Repository](https://github.com/Aider-AI/aider)
- [Tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [NetworkX Documentation](https://networkx.org/)
- [Cursor's Architecture (AI Code Editors)](https://medium.com/aimonks/the-architectures-of-augment-code-cursor-and-windsurf-09aa87e0eb20)
- [Vector-based Code Retrieval](https://milvus.io/blog/why-im-against-claude-codes-grep-only-retrieval-it-just-burns-too-many-tokens.md)
- [Google Cloud Run Best Practices](https://cloud.google.com/run/docs/best-practices)
- [Redis Caching Patterns](https://redis.io/docs/manual/patterns/)

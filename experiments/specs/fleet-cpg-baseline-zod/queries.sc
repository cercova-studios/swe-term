// Golden query set (reduced), Fleet CPG Engine Phase 0 baseline.
// Categories exercised: symbol resolution, reference set, transitive call
// graph, type hierarchy. Each query is timed independently with
// System.nanoTime so JVM/script-compile overhead is not attributed to any
// single query. Emits one JSON line per query to stdout.

import scala.util.{Try, Success, Failure}

def timeIt[A](label: String, category: String)(body: => A): Unit = {
  val start = System.nanoTime()
  val result = Try(body)
  val elapsedMs = (System.nanoTime() - start) / 1000000.0
  val (status, count) = result match {
    case Success(v: Long)      => ("ok", v)
    case Success(v: Int)       => ("ok", v.toLong)
    case Success(v: Iterable[_]) => ("ok", v.size.toLong)
    case Success(_)            => ("ok", -1L)
    case Failure(e)            => ("error:" + e.getMessage, -1L)
  }
  println(s"""{"query_id":"$label","category":"$category","latency_ms":$elapsedMs,"result_count":$count,"status":"$status"}""")
}

@main def exec() = {
  // 1. symbol resolution — does a well-known top-level symbol resolve at all
  timeIt("sym-resolve-object-schema", "symbol_resolution") {
    cpg.method.name("object").l.size
  }
  timeIt("sym-resolve-zodstring-class", "symbol_resolution") {
    cpg.typeDecl.name(".*ZodString.*").l.size
  }

  // 2. reference set — every call site of a widely-used internal function
  timeIt("refs-parse-calls", "reference_set") {
    cpg.call.name("parse").l.size
  }
  timeIt("refs-safe-parse-calls", "reference_set") {
    cpg.call.name("safeParse").l.size
  }

  // 3. transitive call graph (blast radius) — callers/callees of a core method
  timeIt("callgraph-callees-of-parse", "transitive_call_graph") {
    cpg.method.name("parse").l.flatMap(_.callee.name.l).distinct.size
  }
  timeIt("callgraph-callers-of-object", "transitive_call_graph") {
    cpg.method.name("object").l.flatMap(_.caller.name.l).distinct.size
  }

  // 4. type hierarchy — every declared type in the core package
  timeIt("typehier-all-typedecls", "type_hierarchy") {
    cpg.typeDecl.l.size
  }
  timeIt("typehier-zodtype-family", "type_hierarchy") {
    cpg.typeDecl.name(".*Zod.*").l.size
  }
}

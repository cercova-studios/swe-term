// Golden query set (reduced), Fleet CPG Engine Phase 0 baseline — excalidraw.
// Same 4 categories and timing methodology as fleet-cpg-baseline-zod/queries.sc,
// with query targets chosen for excalidraw's actual symbol vocabulary.

import scala.util.{Try, Success, Failure}

def timeIt[A](label: String, category: String)(body: => A): Unit = {
  val start = System.nanoTime()
  val result = Try(body)
  val elapsedMs = (System.nanoTime() - start) / 1000000.0
  val (status, count) = result match {
    case Success(v: Long)        => ("ok", v)
    case Success(v: Int)         => ("ok", v.toLong)
    case Success(v: Iterable[_]) => ("ok", v.size.toLong)
    case Success(_)              => ("ok", -1L)
    case Failure(e)              => ("error:" + e.getMessage, -1L)
  }
  println(s"""{"query_id":"$label","category":"$category","latency_ms":$elapsedMs,"result_count":$count,"status":"$status"}""")
}

@main def exec() = {
  timeIt("sym-resolve-render", "symbol_resolution") {
    cpg.method.name("render").l.size
  }
  timeIt("sym-resolve-scene-class", "symbol_resolution") {
    cpg.typeDecl.name(".*Scene.*").l.size
  }

  timeIt("refs-updatescene-calls", "reference_set") {
    cpg.call.name(".*[Uu]pdateScene.*").l.size
  }
  timeIt("refs-getelementmap-calls", "reference_set") {
    cpg.call.name(".*[Ee]lement.*").l.size
  }

  timeIt("callgraph-callees-of-render", "transitive_call_graph") {
    cpg.method.name("render").l.flatMap(_.callee.name.l).distinct.size
  }
  timeIt("callgraph-callers-of-scene", "transitive_call_graph") {
    cpg.typeDecl.name(".*Scene.*").method.l.flatMap(_.caller.name.l).distinct.size
  }

  timeIt("typehier-all-typedecls", "type_hierarchy") {
    cpg.typeDecl.l.size
  }
  timeIt("typehier-element-family", "type_hierarchy") {
    cpg.typeDecl.name(".*Element.*").l.size
  }
}

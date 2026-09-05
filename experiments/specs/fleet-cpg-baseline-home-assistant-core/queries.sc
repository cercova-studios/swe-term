// Golden query set (reduced), Fleet CPG Engine Phase 0 baseline — home-assistant/core.
// Same 4 categories and timing methodology as fleet-cpg-baseline-zod/queries.sc,
// with query targets chosen for Home Assistant's own vocabulary.

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
  timeIt("sym-resolve-async-setup-entry", "symbol_resolution") {
    cpg.method.name("async_setup_entry").l.size
  }
  timeIt("sym-resolve-homeassistant-class", "symbol_resolution") {
    cpg.typeDecl.name(".*HomeAssistant.*").l.size
  }

  timeIt("refs-async-setup-calls", "reference_set") {
    cpg.call.name("async_setup").l.size
  }
  timeIt("refs-add-entities-calls", "reference_set") {
    cpg.call.name(".*[Aa]dd_entities.*").l.size
  }

  timeIt("callgraph-callees-of-async-setup-entry", "transitive_call_graph") {
    cpg.method.name("async_setup_entry").l.flatMap(_.callee.name.l).distinct.size
  }
  timeIt("callgraph-callers-of-async-setup", "transitive_call_graph") {
    cpg.method.name("async_setup").l.flatMap(_.caller.name.l).distinct.size
  }

  timeIt("typehier-all-typedecls", "type_hierarchy") {
    cpg.typeDecl.l.size
  }
  timeIt("typehier-entity-family", "type_hierarchy") {
    cpg.typeDecl.name(".*Entity.*").l.size
  }
}

// Golden query set (reduced), Fleet CPG Engine Phase 0 baseline — fastapi.
// Same 4 categories and timing methodology as fleet-cpg-baseline-zod/queries.sc,
// with query targets chosen for fastapi's actual symbol vocabulary.

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
  timeIt("sym-resolve-init", "symbol_resolution") {
    cpg.method.name("__init__").l.size
  }
  timeIt("sym-resolve-fastapi-class", "symbol_resolution") {
    cpg.typeDecl.name(".*FastAPI.*").l.size
  }

  timeIt("refs-depends-calls", "reference_set") {
    cpg.call.name("Depends").l.size
  }
  timeIt("refs-jsonresponse-calls", "reference_set") {
    cpg.call.name(".*Response.*").l.size
  }

  timeIt("callgraph-callees-of-init", "transitive_call_graph") {
    cpg.method.name("__init__").l.flatMap(_.callee.name.l).distinct.size
  }
  timeIt("callgraph-callers-of-depends", "transitive_call_graph") {
    cpg.method.name("Depends").l.flatMap(_.caller.name.l).distinct.size
  }

  timeIt("typehier-all-typedecls", "type_hierarchy") {
    cpg.typeDecl.l.size
  }
  timeIt("typehier-router-family", "type_hierarchy") {
    cpg.typeDecl.name(".*Router.*").l.size
  }
}

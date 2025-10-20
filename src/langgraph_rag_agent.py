# langgraph_rag_agent.py
# Deps: pip install langgraph langchain-core pyyaml pandas scikit-learn numpy joblib
from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
import argparse, json
import pandas as pd
import os

# Keep track of root directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

from rag_mapper import (
    load_yaml_corpus,
    build_column_descriptor,
    RAGMapper,
    save_index,
    load_index,
)

# TypedDict defining the runtime state for the RAG agent.
# Tracks execution mode ("index" or "suggest"), file paths, parameters,
# and results during processing. Used for consistent state management
# across agent operations.
class AgentState(TypedDict):
    mode: str                       # "index" or "suggest"
    yaml_dir: Optional[str]
    model_path: str
    csv_path: Optional[str]
    delimiter: str
    k: int
    sample_rows: int
    restrict_to_retrieved: bool
    result: Dict[str, Any]
    log: str

# Validate that the AgentState configuration is consistent with its mode.
# Ensures "index" mode has a YAML directory and "suggest" mode has a CSV path.
# Raises ValueError on invalid or incomplete state; otherwise returns it unchanged.
def validate(state: AgentState) -> AgentState:
    if state["mode"] not in {"index", "suggest"}:
        raise ValueError("mode must be 'index' or 'suggest'")
    if state["mode"] == "index" and not state.get("yaml_dir"):
        raise ValueError("yaml_dir required for index")
    if state["mode"] == "suggest" and not state.get("csv_path"):
        raise ValueError("csv_path required for suggest")
    return state

# Build and save a RAG index from YAML mappings.
# Loads text-label pairs, trains a RAGMapper, serializes the model to disk,
# and updates AgentState with output metadata (model path, counts, log status).
# Returns the updated state.
def node_index(state: AgentState) -> AgentState:
    docs, labels = load_yaml_corpus(state["yaml_dir"])
    mapper = RAGMapper().fit(docs, labels)
    out_path = save_index(mapper, state["model_path"])
    state["result"] = {"model_path": out_path, "unique_targets": len(set(labels)), "kb_items": len(docs)}
    state["log"] = "indexed"
    return state

# Generate mapping suggestions for each CSV column using a trained RAG index.
# Loads the saved RAGMapper, builds text descriptors for columns, queries the model,
# and stores ranked target suggestions with scores in AgentState["result"].
# Updates log status and returns the modified state.
def node_suggest(state: AgentState) -> AgentState:
    mapper = load_index(state["model_path"])
    df = pd.read_csv(state["csv_path"], sep=state["delimiter"])
    res = {}
    for col in df.columns:
        desc = build_column_descriptor(col, df[col], sample_rows=state["sample_rows"])
        sugg = mapper.suggest(desc, k_return=state["k"], restrict_to_retrieved=state["restrict_to_retrieved"])
        res[col] = [
            {"target": t, "score": float(s), "retrieval_sim": float(r), "clf_prob": float(c)}
            for (t, s, r, c) in sugg
        ]
    state["result"] = res
    state["log"] = "suggested"
    return state

# Merge new YAML mappings into an existing RAG index.
# Loads the previous model, appends new docs and labels, retrains a fresh RAGMapper,
# saves the updated index, and records merge statistics in AgentState.
# Returns the updated state.
def node_merge(state: AgentState) -> AgentState:
    old = load_index(state["model_path"])
    new_docs, new_labels = load_yaml_corpus(state["yaml_dir"])
    docs = old._kb_docs + new_docs
    labels = old._kb_labels + new_labels
    mapper = RAGMapper().fit(docs, labels)
    save_index(mapper, state["model_path"])
    state["result"] = {"merged": len(new_docs), "total": len(docs)}
    state["log"] = "merged"
    return state

# Simple routing function that selects the next processing node name
# based on the AgentState mode. Returns "index" or "suggest".
def router(state: AgentState) -> str:
    return "index" if state["mode"] == "index" else "suggest"

# Construct and compile the RAG agent workflow graph.
# Defines processing nodes (validate, index, suggest, merge) and control flow:
# START → validate → (index or suggest via router) → END.
# Returns the compiled StateGraph app ready for execution.
def build_app(state_schema=AgentState) -> Any:
    graph = StateGraph(state_schema)
    graph.add_node("index", node_index)
    graph.add_node("suggest", node_suggest)
    graph.add_node("validate", validate)
    graph.add_node("merge", node_merge)
    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", router, {"index": "index", "suggest": "suggest"})
    graph.add_edge("index", END)
    graph.add_edge("suggest", END)
    return graph.compile()

# Render and save a visual diagram of the compiled LangGraph app.
# Generates a Mermaid-based PNG showing node and edge structure,
# writes it to the specified output directory, and returns the file path.
def export_app_diagram(app, out_dir, basename="langgraph_agent"):
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{basename}.png")
    with open(png_path, "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    return png_path

# --- CLI wrapper ---
def main():
    app = build_app()
    print(app.get_graph().draw_ascii())
    png_path = export_app_diagram(app, os.path.join(BASE_DIR, "outputs"))
    print(f"wrote: {png_path}")
    
    p = argparse.ArgumentParser(description="LangGraph RAG agent (local, pure Python)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("index", help="Index YAML mappings")
    a.add_argument("--yaml_dir", required=True)
    a.add_argument("--model", default="rag_index.npz")

    b = sub.add_parser("suggest", help="Suggest mappings for a CSV")
    b.add_argument("--model", default="rag_index.npz")
    b.add_argument("--csv", required=True)
    b.add_argument("--delimiter", default=",")
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--sample_rows", type=int, default=200)
    b.add_argument("--no_restrict", action="store_true")

    args = p.parse_args()
    if args.cmd == "index":
        state = {
            "mode": "index",
            "yaml_dir": args.yaml_dir,
            "model_path": args.model,
            "csv_path": None,
            "delimiter": ",",
            "k": 5,
            "sample_rows": 200,
            "restrict_to_retrieved": True,
            "result": {},
            "log": ""
        }
    else:
        state = {
            "mode": "suggest",
            "yaml_dir": None,
            "model_path": args.model,
            "csv_path": args.csv,
            "delimiter": args.delimiter,
            "k": args.k,
            "sample_rows": args.sample_rows,
            "restrict_to_retrieved": not args.no_restrict,
            "result": {},
            "log": ""
        }
    out = app.invoke(state)
    print(json.dumps(out["result"], indent=2))

if __name__ == "__main__":
    main()

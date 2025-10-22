# api_runner.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import os

from langgraph_rag_agent import (
    build_app, export_app_diagram, BASE_DIR,
    node_apply_human_decisions,  # direct call for follow-up
)
# Import from your module where these are defined
# from langgraph_rag_agent import (
#     build_app, export_app_diagram, BASE_DIR,
#     node_apply_human_decisions,  # direct call for follow-up
# )

def run_agent(
    mode: str,
    *,
    yaml_dir: Optional[str] = None,
    model_path: str = "rag_index.npz",
    csv_path: Optional[str] = None,
    delimiter: str = ",",
    k: int = 5,
    sample_rows: int = 200,
    restrict_to_retrieved: bool = True,
    write_diagram: bool = False,
    outputs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Programmatic entrypoint. Mirrors CLI states and executes the compiled graph once.
    Returns the full state dict the graph ended with.
    """
    app = build_app()

    if write_diagram:
        out_dir = outputs_dir or os.path.join(BASE_DIR, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        export_app_diagram(app, out_dir)

    if mode not in {"index", "suggest"}:
        raise ValueError("mode must be 'index' or 'suggest'")

    if mode == "index":
        if not yaml_dir:
            raise ValueError("yaml_dir is required for mode='index'")
        state: Dict[str, Any] = {
            "mode": "index",
            "yaml_dir": yaml_dir,
            "model_path": model_path,
            "csv_path": None,
            "delimiter": ",",
            "k": 5,
            "sample_rows": 200,
            "restrict_to_retrieved": True,
            "result": {},
            "log": "",
        }
    else:
        if not csv_path:
            raise ValueError("csv_path is required for mode='suggest'")
        state = {
            "mode": "suggest",
            "yaml_dir": None,
            "model_path": model_path,
            "csv_path": csv_path,
            "delimiter": delimiter,
            "k": k,
            "sample_rows": sample_rows,
            "restrict_to_retrieved": restrict_to_retrieved,
            "result": {},
            "log": "",
        }

    # Single pass through the graph
    out = app.invoke(state)
    return out  # contains keys: mode, result, log, etc.


def index_yaml(
    yaml_dir: str,
    *,
    model_path: str = "rag_index.npz",
    write_diagram: bool = False,
    outputs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for indexing."""
    return run_agent(
        "index",
        yaml_dir=yaml_dir,
        model_path=model_path,
        write_diagram=write_diagram,
        outputs_dir=outputs_dir,
    )


def suggest_mappings(
    csv_path: str,
    *,
    model_path: str = "rag_index.npz",
    delimiter: str = ",",
    k: int = 5,
    sample_rows: int = 200,
    restrict_to_retrieved: bool = True,
    write_diagram: bool = False,
    outputs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs: suggest -> validate_suggestions -> (done | human_review -> apply_human_decisions)
    If decisions file is not present, final state.log == 'waiting_for_decision' and result.review_id is set.
    """
    return run_agent(
        "suggest",
        csv_path=csv_path,
        model_path=model_path,
        delimiter=delimiter,
        k=k,
        sample_rows=sample_rows,
        restrict_to_retrieved=restrict_to_retrieved,
        write_diagram=write_diagram,
        outputs_dir=outputs_dir,
    )


def apply_decisions(review_id: str) -> Dict[str, Any]:
    """
    Apply human decisions without re-running suggestion.
    Calls node_apply_human_decisions directly with a minimal state carrying the review_id.
    Returns the updated state with post_validation.accepted/issues and log='human_review_applied'
    or log='waiting_for_decision' if the decision file is still absent.
    """
    # Build a minimal state expected by node_apply_human_decisions
    state = {
        "mode": "suggest",           # mode is irrelevant here but validated type requires it
        "yaml_dir": None,
        "model_path": "rag_index.npz",
        "csv_path": None,
        "delimiter": ",",
        "k": 5,
        "sample_rows": 200,
        "restrict_to_retrieved": True,
        "result": {"review_id": review_id},
        "log": "",
    }
    return node_apply_human_decisions(state)

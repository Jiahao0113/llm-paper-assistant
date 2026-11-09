"""All research systems by name (the LangGraph one lives in its own module, which imports research.py)."""
from .research import SYSTEMS as BASE
from .research import staged, staged_noverify
from .research_graph import staged_graph

SYSTEMS = {**BASE, "staged_noverify": staged_noverify, "graph": staged_graph,
           "staged_r1": lambda r, task: staged(r, task, max_repairs=1)}  # one repair round instead of two

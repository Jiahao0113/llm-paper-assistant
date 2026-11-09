"""All research systems by name (the LangGraph one lives in its own module, which imports research.py)."""
from .research import SYSTEMS as BASE
from .research import staged_noverify
from .research_graph import staged_graph

SYSTEMS = {**BASE, "staged_noverify": staged_noverify, "graph": staged_graph}

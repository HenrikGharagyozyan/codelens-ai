"""The single Rich console used across the CLI and the indexer.

Sharing one instance keeps output width, colour detection and live status
displays consistent, and stops nested progress widgets fighting each other.
"""

from rich.console import Console

console = Console()

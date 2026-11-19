"""Small standalone helpers with no dependency on `src.agents` / `src.services`.

Deliberately kept dependency-free: both `src/agents/__init__.py` and `src/services/__init__.py`
eagerly import their submodules, so a helper needed by BOTH packages (like `prompt_loader`) must
live outside either package — importing it from inside `src.agents` or `src.services` would
re-trigger that package's `__init__.py` mid-import and cause a circular import.
"""

"""
Isolated import helper for baseline repos.

Each cloned repo has its own ``utils.py``, ``models/``, etc. that collide
with the project-root packages when both are on ``sys.path``.

Strategy: temporarily hide conflicting modules from sys.modules,
import the target, then restore. The repo stays on sys.path because
some modules need it for lazy/transitive imports at runtime.
"""

import sys
import importlib

# Module names that commonly clash between repos and the project root.
_CONFLICT_PREFIXES = (
    'utils', 'models', 'tasks', 'lib', 'collator', 'augmentations',
    'datautils', 'config',
)


def isolated_import(repo_dir, module_name):
    """Import *module_name* from *repo_dir*.

    Returns the imported module object.
    """
    # 1. Save and remove conflicting modules from sys.modules
    saved = {}
    for name in list(sys.modules):
        for prefix in _CONFLICT_PREFIXES:
            if name == prefix or name.startswith(prefix + '.'):
                saved[name] = sys.modules.pop(name)
                break

    # 2. Also remove any cached version of the target module
    #    (e.g. 'infots' cached as baselines.infots namespace)
    bare = module_name.split('.')[0]
    for name in list(sys.modules):
        if name == bare or name == f'baselines.{bare}' or name.startswith(bare + '.'):
            if name not in saved:
                saved[name] = sys.modules.pop(name)

    # 3. Put repo_dir at front of sys.path
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    try:
        mod = importlib.import_module(module_name)
    finally:
        # 4. Clean up repo-specific conflict modules from sys.modules
        #    but keep the target module and its submodules (they're needed)
        for name in list(sys.modules):
            for prefix in _CONFLICT_PREFIXES:
                if name == prefix or name.startswith(prefix + '.'):
                    sys.modules.pop(name, None)
                    break

        # 5. Restore the project's original modules
        sys.modules.update(saved)

    # NOTE: repo_dir stays on sys.path so transitive imports work at runtime.
    return mod

#!/usr/bin/env python3
"""Explore the a2a package structure"""

import pkgutil
import a2a

print("=== A2A Package Structure ===")
print(f"a2a path: {a2a.__path__}")
print(f"a2a file: {a2a.__file__}")
print()

print("Available modules in a2a:")
for importer, modname, ispkg in pkgutil.walk_packages(
    path=a2a.__path__, 
    prefix='a2a.', 
    onerror=lambda x: None
):
    print(f"  {'[PKG]' if ispkg else '[MOD]'} {modname}")

print("\nTrying to import common patterns:")
try:
    from a2a.types import AgentCard
    print("✓ from a2a.types import AgentCard")
except ImportError as e:
    print(f"✗ from a2a.types import AgentCard - {e}")

try:
    from a2a.client import A2AClient
    print("✓ from a2a.client import A2AClient")
except ImportError as e:
    print(f"✗ from a2a.client import A2AClient - {e}")

try:
    from a2a.decorators import skill
    print("✓ from a2a.decorators import skill")
except ImportError as e:
    print(f"✗ from a2a.decorators import skill - {e}")

try:
    from a2a import Agent
    print("✓ from a2a import Agent")
except ImportError as e:
    print(f"✗ from a2a import Agent - {e}")

try:
    from a2a.agent import Agent
    print("✓ from a2a.agent import Agent")
except ImportError as e:
    print(f"✗ from a2a.agent import Agent - {e}")

print("\nChecking a2a.types contents:")
try:
    from a2a import types
    print("types module contents:", [x for x in dir(types) if not x.startswith('_')])
except ImportError as e:
    print(f"Cannot import a2a.types: {e}")

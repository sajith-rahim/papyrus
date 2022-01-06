import sys
from pathlib import Path
from shutil import copytree, ignore_patterns


# This script initializes new pytorch project with the template files.
# Run `python3 spawn.py ../new-project-dir new project` then new project named

current_dir = Path()
assert (current_dir / 'spawn.py').is_file(), 'Script should be executed in the root directory'
assert len(sys.argv) == 2

project_name = Path(sys.argv[1])
target_dir = current_dir / project_name

ignore = [".git", "data", "outputs", "spawn.py", "LICENSE", "venv", "__pycache__", "logs"]
copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore))
print(
    """
─────────────╔═╗
╔═╦═╗╔═╦╦╦╦╦╦╣═╣
║╬║╬╚╣╬║║║╔╣║╠═║
║╔╩══╣╔╬╗╠╝╚═╩═╝
╚╝───╚╝╚═╝
    """)
print('New project initialized at', target_dir.absolute().resolve())
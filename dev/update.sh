#!/bin/sh
set -e

PKG="quickdistill"

if [ ! -f "pyproject.toml" ]; then
  echo "pyproject.toml not found"
  exit 1
fi

BUMP="${1:-patch}"

# read current version
CURR="$(python - <<'PY'
import re
t=open("pyproject.toml").read()
m=re.search(r'(?m)^\s*version\s*=\s*"(.*?)"\s*', t)
print(m.group(1) if m else "")
PY
)"
[ -n "$CURR" ] || { echo "could not read current version"; exit 1; }

# compute new version
new_version() {
  v="$1"
  IFS='.' read -r MA MI PA <<EOF
$v
EOF
  MA=${MA:-0}; MI=${MI:-0}; PA=${PA:-0}
  case "$2" in
    patch) PA=$((PA+1));;
    minor) MI=$((MI+1)); PA=0;;
    major) MA=$((MA+1)); MI=0; PA=0;;
    *) echo "$2"; return;;
  esac
  echo "${MA}.${MI}.${PA}"
}

if echo "$BUMP" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+'; then
  NEW="$BUMP"
else
  NEW="$(new_version "$CURR" "$BUMP")"
fi

echo "Bumping version: $CURR -> $NEW"

# update version in pyproject.toml
NEW_VER="$NEW" python - <<'PY'
import os, re, sys
p="pyproject.toml"
new=os.environ["NEW_VER"]
t=open(p).read()
m=re.search(r'(?m)^(\s*version\s*=\s*")([^"]*)(")', t)
if not m:
  print("version key not found"); sys.exit(1)
t = t[:m.start(2)] + new + t[m.end(2):]
open(p,"w").write(t)
print(f"✓ Updated pyproject.toml: {m.group(2)} -> {new}")
PY

# update version in __init__.py
NEW_VER="$NEW" python - <<'PY'
import os, re, sys
p="quickdistill/__init__.py"
new=os.environ["NEW_VER"]
t=open(p).read()
m=re.search(r'(?m)^(__version__\s*=\s*")([^"]*)(")', t)
if not m:
  print("__version__ not found"); sys.exit(1)
t = t[:m.start(2)] + new + t[m.end(2):]
open(p,"w").write(t)
print(f"✓ Updated __init__.py: {m.group(2)} -> {new}")
PY

# clean old artifacts
echo "🧹 Cleaning old build artifacts..."
rm -rf build dist *.egg-info
find . -maxdepth 1 -name "*.egg-info" -exec rm -rf {} \;

# rebuild
echo "📦 Building package..."
python -m pip install -U build twine
python -m build

echo "✓ Build complete!"
echo ""
echo "To publish to PyPI, run:"
echo "  python -m twine upload dist/*"
echo ""
echo "Or set PYPI_TOKEN and run:"
echo "  python -m twine upload -u __token__ -p \$PYPI_TOKEN dist/*"

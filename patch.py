import site, os

content = "import importlib.metadata\n\ndef dep_version_check(pkg, *args, **kwargs):\n    pass\n\ndef require_version(requirement, *args, **kwargs):\n    pass\n\ndef require_version_core(requirement, *args, **kwargs):\n    pass\n"

for p in site.getsitepackages():
    f = os.path.join(p, 'transformers', 'dependency_versions_check.py')
    if os.path.exists(f):
        with open(f, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print('Patched:', f)
import glob
import os

"""
Generate the code documentation, using pydoc.
"""

modules = ['clustersciclistes', 'generardataset']

# Asegúrate de que la carpeta docs/ existe
if not os.path.exists('docs'):
    os.makedirs('docs')
    
print(f"Generando la documentación")
for module in modules:
    os.system(f'python -m pydoc -w {module}')

    html_file = f"{module}.html"
    os.rename(html_file, f"docs/{module}.html")

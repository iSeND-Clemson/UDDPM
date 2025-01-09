# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:29:00 2025

@author: MaxGr
"""

import os
import pkgutil
import importlib
import subprocess

def find_project_dependencies(project_path):
    """
    Scans a Python project to find all directly imported packages.

    Args:
        project_path (str): Path to the root of the project.

    Returns:
        set: A set of unique package names directly imported in the project.
    """
    dependencies = set()

    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        parts = line.split()
                        if line.startswith('import '):
                            dependencies.add(parts[1].split(',')[0].split('.')[0])
                        elif line.startswith('from '):
                            dependencies.add(parts[1].split(',')[0].split('.')[0])

    # Filter out standard library modules
    standard_libs = {name for _, name, _ in pkgutil.iter_modules() if importlib.util.find_spec(name) is None}
    return {dep.strip(';').strip() for dep in dependencies} - standard_libs


def get_installed_versions(dependencies):
    """
    Retrieves the installed versions of the given dependencies.

    Args:
        dependencies (set): A set of unique package names.

    Returns:
        dict: A dictionary with package names as keys and their versions as values.
    """
    package_versions = {}
    for package in dependencies:
        try:
            # Use pip show to get package version
            result = subprocess.run(["pip", "show", package], capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    version = line.split()[1]
                    package_versions[package] = version
                    print(f"{package}=={version}")
                    break
                
        except subprocess.CalledProcessError:
            # package_versions[package] = "Unknown"
            print(package)
    return package_versions


def save_requirements(package_versions, output_file='requirements.txt'):
    """
    Saves the list of dependencies with versions to a requirements.txt file.

    Args:
        package_versions (dict): A dictionary with package names and their versions.
        output_file (str): The file to save the requirements to.
    """
    with open(output_file, 'w') as f:
        for package, version in sorted(package_versions.items()):
            if version != "Unknown":
                f.write(f"{package}=={version}\n")
            else:
                f.write(f"{package}\n")

    print(f"Requirements saved to {output_file}")


if __name__ == '__main__':
    project_path = '.'  # Specify your project root path
    dependencies = find_project_dependencies(project_path)
    package_versions = get_installed_versions(dependencies)
    save_requirements(package_versions)




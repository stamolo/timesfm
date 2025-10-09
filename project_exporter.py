#!/usr/bin/env python3
# project_exporter.py — Export a project's structure and file contents into a single text file.

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Iterable, Set

# --- configuration ---------------------------------------------------------
# «Белый» список исключений: показываем в структуре, но не заходим внутрь и не включаем в контент
DEFAULT_EXCLUDES: Set[str] = {

    '.env',
    'output'


}

# Чёрный список: полностью игнорируемые элементы
DEFAULT_BLACKLIST: Set[str] = {
    '.venv', '.idea', 'package-lock.json', 'node_modules', 'readme.md', 'templates',
    'frontend', '.pytest_cache', '.git', '__pycache__', 'reports_ide',
    'tests', 'venv', 'lib', 'technical_description',
    'uni.rar', 'helpers', 'requirements.txt', 'readme_comm.md',
    'pytest.ini', '!labeled_output__.xlsx',
    # 'Dockerfile.prod', 'Dockerfile',
    # 'docker-compose.yml', 'docker-compose.prod.yml',
    'tmp', 'test_sps', 'dataset', 'pipeline.log', 'saved_model',
     'pred_web.py', 'test.py', 'train_web.py',
     'utils', 'main_1.py', 'pred256_3.py',

    'step_1_extraction.py',
    'step_2_padding.py', 'step_3_prediction.py', 'step_4_remapping.py', 'step_5_tool_depth.py', 'step_6_block_average.py',
    'step_7_advanced_reset.py', 'step_8_bottom_hole_depth.py', 'step_9_derivatives.py',
    #'step_10_above_bottom_hole.py',
    #'step_11_anomaly_detection.py',
    #'step_12_plotting.py'

}

DEFAULT_MAX_BYTES = 1_000_000  # 1 MiB
STRUCTURE_HEADER = "=== PROJECT STRUCTURE ==="
CONTENT_HEADER   = "=== FILE CONTENTS ==="

# --- helpers ---------------------------------------------------------------

def is_binary(path: Path, sniff: int = 1024) -> bool:
    """Heuristically detect binary files by looking for NUL bytes."""
    try:
        with path.open('rb') as fp:
            return b'\0' in fp.read(sniff)
    except Exception:
        return True

def is_ignored(
    relative_path_str: str,
    component_ignore_list: Set[str],
    fullpath_ignore_list: Set[str]
) -> bool:
    """
    Оптимизированная проверка, следует ли игнорировать путь.
    """
    # 1. Проверка на полное совпадение пути
    if relative_path_str in fullpath_ignore_list:
        return True
    # 2. Проверка на совпадение по компонентам
    # (создание Path объекта здесь - основная нагрузка, но это необходимо)
    path_components = Path(relative_path_str).parts
    for component in path_components:
        if component in component_ignore_list:
            return True
    return False

def dump_project(
    root: Path,
    out_file: Path,
    max_bytes: int,
    skip_binary: bool,
    excludes: Set[str],
    blacklist: Set[str],
):
    """Записывает в out_file структуру и содержимое проекта за один проход."""
    root = root.resolve()

    # --- Оптимизация: разделяем списки на компоненты и полные пути ---
    blacklist_components = {item for item in blacklist if '/' not in item and '\\' not in item}
    blacklist_fullpaths = blacklist - blacklist_components
    excludes_components = {item for item in excludes if '/' not in item and '\\' not in item}
    excludes_fullpaths = excludes - excludes_components

    structure_lines: List[str] = []
    content_paths: List[Path] = []

    def traverse_tree(current_path: Path, indent_level: int):
        """
        Рекурсивно обходит дерево, одновременно собирая структуру и пути к файлам.
        """
        # Сортируем для консистентного вывода: сначала папки, потом файлы
        try:
            items = sorted(current_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            items = [] # Пропускаем недоступные директории

        for item in items:
            rel_path_str = item.relative_to(root).as_posix()
            indent = '    ' * indent_level

            if is_ignored(rel_path_str, blacklist_components, blacklist_fullpaths):
                continue

            is_excluded = is_ignored(rel_path_str, excludes_components, excludes_fullpaths)

            if item.is_dir():
                structure_lines.append(f"{indent}{item.name}/")
                if not is_excluded:
                    traverse_tree(item, indent_level + 1)
            else: # Это файл
                structure_lines.append(f"{indent}{item.name}")
                if not is_excluded:
                    content_paths.append(item)

    # --- Основной однопроходный алгоритм ---
    structure_lines.append("./")
    traverse_tree(root, 1) # Начинаем обход с корневой директории

    with out_file.open('w', encoding='utf-8', errors='replace') as out:
        # --- Записываем структуру проекта ---
        out.write(f"{STRUCTURE_HEADER}\n")
        out.write("\n".join(structure_lines))
        out.write("\n\n\n")

        # --- Записываем содержимое файлов ---
        out.write(f"{CONTENT_HEADER}\n")
        for path in content_paths:
            # 1) Пропускаем сам выходной файл (дополнительная проверка)
            if path.resolve() == out_file.resolve():
                continue
            # 2) Ограничение по размеру
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except Exception:
                continue
            # 3) Бинарные файлы (опционально)
            if skip_binary and is_binary(path):
                continue

            rel_str = path.relative_to(root).as_posix()
            out.write(f"### {rel_str}\n")
            try:
                out.write(path.read_text(encoding='utf-8', errors='replace'))
            except Exception as exc:
                out.write(f"<error reading file: {exc}>\n")
            out.write("\n\n")

# --- CLI -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a project's structure and file contents into a single text file."
    )
    parser.add_argument('project_root', type=Path, help='Path to the project directory')
    parser.add_argument('-o', '--output', type=Path, default=Path('project_dump.txt'),
                        help='Output file path')
    parser.add_argument('--max-bytes', type=int, default=DEFAULT_MAX_BYTES,
                        help='Skip files larger than this many bytes')
    parser.add_argument('--skip-binary', action='store_true',
                        help='Skip files detected as binary')
    parser.add_argument('--exclude', nargs='*', default=[], metavar='NAME',
                        help='Дополнительные директории/файлы, которые выводить в структуре, но не включать в контент')
    return parser.parse_args()

def main():
    args = parse_args()
    script_name = Path(__file__).name

    # Добавляем сам скрипт и выходной файл в черный список, чтобы они не попали в вывод
    all_blacklist = DEFAULT_BLACKLIST.union({script_name, args.output.name})
    all_excludes = DEFAULT_EXCLUDES.union(args.exclude)

    dump_project(
        args.project_root,
        args.output,
        args.max_bytes,
        args.skip_binary,
        all_excludes,
        all_blacklist,
    )
    print(f"Exported project to {args.output}")

if __name__ == '__main__':
    main()
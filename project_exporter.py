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
    # 'README.md',
    'dv_record_table',
    # 'dv_core/schema_version',
    # 'dv_core/domain',
    # 'dv_core/repos',
    # 'dv_core/utils',
    # 'dv_core/quantities',
    '.env',
    # '.env.example',
    # 'package-lock.json'
}

# Чёрный список: полностью игнорируемые элементы
DEFAULT_BLACKLIST: Set[str] = {
    '.venv', '.idea', 'package-lock.json', 'node_modules',
    'frontend', '.pytest_cache', '.git', '__pycache__', 'reports_ide',
    'dv_core/pyproject.toml', 'dv_core/uow.py',
    'tests', 'venv', 'lib', 'technical_description',
    'uni.rar', 'helpers', 'requirements.txt', 'readme_comm.md',
    'pytest.ini',
    # 'Dockerfile.prod', 'Dockerfile',
    # 'docker-compose.yml', 'docker-compose.prod.yml',
    'tmp', 'test_sps', 'dataset'
}

DEFAULT_MAX_BYTES = 1_000_000  # 1 MiB
STRUCTURE_HEADER = "=== PROJECT STRUCTURE ==="
CONTENT_HEADER = "=== FILE CONTENTS ==="


# --- helpers ---------------------------------------------------------------

def is_binary(path: Path, sniff: int = 1024) -> bool:
    """Heuristically detect binary files by looking for NUL bytes."""
    try:
        with path.open('rb') as fp:
            return b'\0' in fp.read(sniff)
    except Exception:
        return True


def path_part_in_rules(path: Path, root: Path, rules: Set[str]) -> bool:
    """
    Проверяет, есть ли хотя бы одна часть относительного пути (папка или файл)
    в наборе правил rules. Сравнение идет на точное совпадение.
    """
    try:
        # Получаем части пути относительно корня проекта
        # Например, для 'utils/create_dataset.py' это будет ('utils', 'create_dataset.py')
        relative_parts = path.relative_to(root).parts
        # Проверяем пересечение множества частей пути и множества правил
        return not rules.isdisjoint(relative_parts)
    except ValueError:
        return False


def iter_structure(
        dir_path: Path,
        root: Path,
        blacklist: Set[str],
        excludes: Set[str],
        indent_level: int = 0
) -> Iterable[str]:
    """
    Рекурсивно выводит дерево, но пропускает целиком любые
    папки/файлы, чьи имена точно совпадают с элементами в blacklist.
    """
    # если этот путь (или его родительская папка) в чёрном списке — сразу выходим
    if path_part_in_rules(dir_path, root, blacklist):
        return

    indent = '    ' * indent_level
    name = dir_path.name if indent_level else '.'
    yield f"{indent}{name}/"

    for item in sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        # пропускаем по чёрному списку (точное совпадение имени)
        if path_part_in_rules(item, root, blacklist):
            continue

        # Для файлов (не директорий) проверяем список исключений
        if not item.is_dir() and path_part_in_rules(item, root, excludes):
            continue

        if item.is_dir():
            yield from iter_structure(item, root, blacklist, excludes, indent_level + 1)
        else:
            yield f"{indent}    {item.name}"


def dump_project(
        root: Path,
        out_file: Path,
        max_bytes: int,
        skip_binary: bool,
        excludes: Set[str],
        blacklist: Set[str],
):
    """Записывает в out_file структуру и содержимое проекта."""
    root = root.resolve()
    with out_file.open('w', encoding='utf-8', errors='replace') as out:
        # --- PROJECT STRUCTURE ---
        out.write(f"{STRUCTURE_HEADER}\n")
        for line in iter_structure(root, root, blacklist, excludes):
            out.write(line + "\n")
        out.write("\n\n")

        # --- FILE CONTENTS ---
        out.write(f"{CONTENT_HEADER}\n")
        for path in sorted(root.rglob('*'), key=lambda p: (p.is_file(), str(p).lower())):
            # 1) полностью игнорируем, если часть пути в чёрном списке
            if path_part_in_rules(path, root, blacklist):
                continue
            # 2) пропускаем каталоги
            if path.is_dir():
                continue
            # 3) если часть пути в списке excludes — не пишем контент
            if path_part_in_rules(path, root, excludes):
                continue
            # 4) ограничение по размеру
            try:
                size = path.stat().st_size
            except Exception:
                continue
            if size > max_bytes:
                continue
            # 5) бинарные (опционально)
            if skip_binary and is_binary(path):
                continue

            rel = path.relative_to(root).as_posix()
            out.write(f"### {rel}\n")
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
    output_name = args.output.name

    # Собираем списки: в excludes попадают DEFAULT_EXCLUDES + пользовательские + сам скрипт и выходной файл
    all_excludes = DEFAULT_EXCLUDES.union(args.exclude, {script_name, output_name})
    all_blacklist = DEFAULT_BLACKLIST

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

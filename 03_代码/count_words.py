"""逐文件统计 Markdown 论文正文的估算字数。

口径：每个汉字计 1 字，连续的英文或数字计 1 词；计入标题、正文和表格内容，
不计 Markdown 标记、图片路径、链接地址、HTML 注释及 LaTeX 公式。
这不是 Word 或学校排版系统的官方字数。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_DIRECTORY = Path(__file__).absolute().parents[1] / "06_论文正文"
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
DISPLAY_MATH = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)|\\\(.*?\\\)", re.DOTALL)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
REFERENCE_MARKER = re.compile(r"\[R\d+\]")
URL = re.compile(r"https?://\S+")
WORD = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]|[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*")


def count_words(markdown: str) -> int:
    """按脚本开头说明的口径统计可见文字。"""
    text = HTML_COMMENT.sub(" ", markdown)
    text = DISPLAY_MATH.sub(" ", text)
    text = INLINE_MATH.sub(" ", text)
    text = IMAGE.sub(" ", text)
    text = LINK.sub(r"\1", text)
    text = REFERENCE_MARKER.sub(" ", text)
    text = URL.sub(" ", text)
    return len(WORD.findall(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help=f"Markdown 所在目录，默认：{DEFAULT_DIRECTORY}",
    )
    args = parser.parse_args()
    directory = args.directory.expanduser().absolute()
    if not directory.is_dir():
        parser.error(f"目录不存在：{directory}")

    files = sorted(directory.rglob("*.md"), key=lambda path: str(path.relative_to(directory)))
    if not files:
        parser.error(f"目录中没有 .md 文件：{directory}")

    results = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            print(f"无法读取 {path}：{exc}", file=sys.stderr)
            return 1
        results.append((str(path.relative_to(directory)), count_words(content)))

    name_width = max(len("文件"), *(len(name) for name, _ in results))
    print(f"统计目录：{directory}")
    print("统计口径：汉字每字计 1，连续英文或数字计 1；不计 Markdown 标记及公式")
    print(f"{'文件':<{name_width}}  {'字数':>8}")
    print(f"{'-' * name_width}  {'-' * 8}")
    for name, count in results:
        print(f"{name:<{name_width}}  {count:>8,}")
    print(f"{'-' * name_width}  {'-' * 8}")
    print(f"{'合计':<{name_width}}  {sum(count for _, count in results):>8,}")
    print(f"文件数：{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

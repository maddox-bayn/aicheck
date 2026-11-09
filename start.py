"""Create a checkpoint workspace without exposing reference implementations."""

import argparse
import ast
from pathlib import Path

from check import ROOT, read_checkpoint


def skeleton(source):
    tree = ast.parse(source)
    stub = ast.parse('raise NotImplementedError("Implement this exercise")').body
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = stub
        elif isinstance(node, ast.ClassDef):
            methods = [
                m
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for method in methods:
                method.body = stub
            node.body = methods or [ast.Pass()]
    return ast.unparse(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=int, choices=range(1, 21))
    parser.add_argument("--directory", type=Path, default=Path("submissions"))
    args = parser.parse_args()
    folder = args.directory / f"cp{args.checkpoint:02}"
    folder.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "checkpoints" / f"{args.checkpoint:02}.md").read_text()
    descriptions = [
        line.removeprefix("- **Description**: ")
        for line in text.splitlines()
        if line.startswith("- **Description**: ")
    ]
    for index, title, code, _ in read_checkpoint(args.checkpoint):
        path = folder / f"ex{index:02}.py"
        try:
            with path.open("x") as stream:
                stream.write(
                    f"# {args.checkpoint:02}.{index} {title}\n# {descriptions[index - 1]}\n\n"
                )
                stream.write(skeleton(code) + "\n")
            print(f"Created {path}")
        except FileExistsError:
            print(f"Kept {path}")


if __name__ == "__main__":
    main()

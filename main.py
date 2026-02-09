import os.path
import sys

from schema_to_df import parse_pg_dump
from export_to_mermaid import generate_mermaid_erd
import jinja2
from argparse import ArgumentParser
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
parser = ArgumentParser()
parser.add_argument(
    "input_path", type=str,
    help="File path of \"pg_dump\" output, which is PostgreSQL database schema."
)
cmd, _ = parser.parse_known_args()

schema_path = cmd.input_path
base_dir = os.path.dirname(schema_path)
tables = parse_pg_dump(schema_path)
er_mermaid = generate_mermaid_erd(tables)
with open("mermaid.html") as f:
    template_str = f.read()
template = jinja2.Environment().from_string(template_str)
er_html = template.render({'mermaid_code': er_mermaid})
with open(os.path.join(base_dir, "er_diagram.html"), "w") as f:
    f.write(er_html)
with open(os.path.join(base_dir, "er_diagram.md"), "w") as f:
    f.write("```mermaid\n")
    f.write(er_mermaid)
    f.write("\n```")

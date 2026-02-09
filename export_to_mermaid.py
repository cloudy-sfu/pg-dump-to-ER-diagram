import pandas as pd
import re


def sanitize_dtype(dtype: str) -> str:
    """Convert SQL types to single-word Mermaid-safe tokens."""
    if not dtype or dtype == "nan":
        return "_"
    dtype = re.sub(r"\((.*?)\)", r"_\1", dtype)
    dtype = dtype.strip().replace(" ", "_")
    dtype = dtype.replace(".", "_")
    return dtype


def _table_type_label(table_type: str) -> str:
    """Return a parenthetical suffix for non-table entity types."""
    if table_type == "table":
        return ""
    # e.g. "materialized_view" -> "(materialized view)"
    return f" ({table_type.replace('_', ' ')})"


def generate_mermaid_erd(df: pd.DataFrame) -> str:

    df["table_short"] = df["table_name"].str.replace("public.", "", regex=False)
    df["fk_table_short"] = df["fk_table"].str.replace("public.", "", regex=False)
    df["column_src"] = df["column_src"].fillna("")

    # ── helper: resolve table_type per table_short ──────────────────────
    table_type_map: dict[str, str] = (
        df.drop_duplicates(subset="table_short")
        .set_index("table_short")["table_type"]
        .to_dict()
    )

    # ── 1. Build entity blocks ──────────────────────────────────────────
    entities: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        tbl = row["table_short"]
        col = row["column_name"] if pd.notna(row["column_name"]) else "_"
        dtype = sanitize_dtype(str(row["data_type"]))

        markers = []
        if row["is_pk"]:
            markers.append("PK")
        if row["is_fk"]:
            markers.append("FK")
        if row["is_uq"]:
            markers.append("UK")
        marker_str = ",".join(markers)

        req = "*" if row.get("required") else ""
        attr_line = f"    {dtype} {col}{req}"
        if marker_str:
            attr_line += f" {marker_str}"

        entities.setdefault(tbl, []).append(attr_line)

    # ── 2. Collect relationships ────────────────────────────────────────
    relationships: list[tuple[str, str, str, str]] = []
    seen_rels = set()

    fk_rows = df[df["is_fk"] == True]
    for _, row in fk_rows.iterrows():
        src = row["table_short"]
        src_col = row["column_name"]
        tgt = row["fk_table_short"]
        tgt_col = row["fk_col"]
        key = (src, src_col, tgt, tgt_col)
        if key not in seen_rels:
            seen_rels.add(key)
            relationships.append(key)

    mv_rows = df[(df["table_type"] == "materialized_view") & (df["column_src"] != "")]
    for _, row in mv_rows.iterrows():
        src = row["table_short"]
        src_col = row["column_name"] if pd.notna(row["column_name"]) else "_"
        sources = [s.strip() for s in row["column_src"].split(",")]
        for ref in sources:
            parts = ref.replace("public.", "").split(".")
            if len(parts) == 2:
                tgt_table, tgt_col = parts
                key = (src, src_col, tgt_table, tgt_col)
                if key not in seen_rels:
                    seen_rels.add(key)
                    relationships.append(key)

    # ── 3. Render Mermaid ───────────────────────────────────────────────
    lines = ["erDiagram"]

    for tbl, attrs in entities.items():
        suffix = _table_type_label(table_type_map.get(tbl, "table"))
        # Mermaid entity labels use ["Label"] syntax for display names
        if suffix:
            lines.append(f'  {tbl}["{tbl}{suffix}"] {{')
        else:
            lines.append(f"  {tbl} {{")
        for a in attrs:
            lines.append(a)
        lines.append("  }")
        lines.append("")

    for src, src_col, tgt, tgt_col in relationships:
        src_type = table_type_map.get(src, "table")
        connector = "}o..o{" if src_type == "materialized_view" else "}o--||"
        label = f"{src}.{src_col} -> {tgt}.{tgt_col}"
        lines.append(f'  {src} {connector} {tgt} : "{label}"')

    return "\n".join(lines)

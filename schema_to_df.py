import re
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Low-level helpers
# ---------------------------------------------------------------------------

def _read_sql(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = [
            l for l in f
            if not l.strip().startswith("\\")
            and not l.strip().startswith("SET ")
            and not l.strip().startswith("SELECT pg_catalog")
            and not l.strip().startswith("COMMENT ON")
        ]
    return "\n".join(lines)


def _split_statements(sql: str) -> list[str]:
    stmts, buf, depth = [], [], 0
    for ch in sql:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        buf.append(ch)
        if ch == ';' and depth <= 0:
            stmts.append("".join(buf).strip())
            buf = []
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return [s for s in stmts if s]


def _split_top_level(s: str, delim: str = ',') -> list[str]:
    parts, buf, depth = [], [], 0
    in_sq = False
    for ch in s:
        if ch == "'" and not in_sq:
            in_sq = True
        elif ch == "'" and in_sq:
            in_sq = False
        if not in_sq:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
        if ch == delim and depth == 0 and not in_sq:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith('(') and s.endswith(')'):
        depth = 0
        match = False
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and i == len(s) - 1:
                match = True
            elif depth == 0 and i < len(s) - 1:
                break
        if match:
            s = s[1:-1].strip()
        else:
            break
    return s


def _sanitize_pg_operators(sql: str) -> str:
    return re.sub(r'OPERATOR\s*\(\s*[\w.]+\s*\)', '-', sql, flags=re.I)


def _normalize_table_name(name: str) -> str:
    n = name.lower().strip().replace('"', '')
    if n.startswith("public."):
        n = n[len("public."):]
    return n


def _find_keyword_pos(sql: str, keyword: str) -> int:
    pat = re.compile(r'\b' + keyword + r'\b', re.I)
    depth = 0
    in_sq, in_dq = False, False
    for i, ch in enumerate(sql):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
        if depth == 0 and not in_sq and not in_dq:
            m = pat.match(sql, i)
            if m:
                return i
    return -1


# ---------------------------------------------------------------------------
# 2. Parse CREATE TABLE
# ---------------------------------------------------------------------------

_RE_CREATE_TABLE = re.compile(
    r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
    r'(?P<name>[\w."]+)\s*\((?P<body>.*)\)\s*;',
    re.S | re.I,
)

_RE_COL = re.compile(r'^(?P<col>\w+)\s+(?P<rest>.+)$', re.I)
_SKIP_KW = {"PRIMARY", "UNIQUE", "CHECK", "CONSTRAINT", "FOREIGN", "EXCLUDE"}

# Regex to strip trailing column constraints, leaving only the data type.
# Handles: NOT NULL, NULL, DEFAULT ..., REFERENCES ..., CHECK(...),
#          GENERATED ..., COLLATE ..., CONSTRAINT ...
_RE_COL_CONSTRAINT_TAIL = re.compile(
    r'\s+(?:NOT\s+NULL|NULL|DEFAULT\s+.*|REFERENCES\s+.*|CHECK\s*\(.*|'
    r'GENERATED\s+.*|COLLATE\s+.*|CONSTRAINT\s+.*)$',
    re.I,
)


def _extract_data_type(rest: str) -> str:
    """
    Given the portion after 'column_name ' in a CREATE TABLE column definition,
    extract just the data type.

    Examples:
        "character varying(9) NOT NULL"     -> "character varying(9)"
        "jsonb"                             -> "jsonb"
        "public.vector(1536)"              -> "public.vector(1536)"
        "smallint NOT NULL"                 -> "smallint"
        "boolean"                           -> "boolean"
        "character varying(64)[]"           -> "character varying(64)[]"
        "integer DEFAULT 0 NOT NULL"        -> "integer"
    """
    s = rest.strip()
    # Iteratively strip trailing constraints
    while True:
        new_s = _RE_COL_CONSTRAINT_TAIL.sub('', s).strip()
        if new_s == s:
            break
        s = new_s
    return s


def _parse_create_table(stmt: str) -> dict | None:
    m = _RE_CREATE_TABLE.search(stmt)
    if not m:
        return None
    table_name = m.group("name").replace('"', '')
    parts = _split_top_level(m.group("body"), ',')
    columns = []
    for p in parts:
        p_up = p.strip().upper()
        if any(p_up.startswith(kw) for kw in _SKIP_KW):
            continue
        cm = _RE_COL.match(p.strip())
        if not cm:
            continue
        col_name = cm.group("col")
        if col_name.upper() in _SKIP_KW:
            continue
        rest = cm.group("rest")
        not_null = bool(re.search(r'\bNOT\s+NULL\b', rest, re.I))
        data_type = _extract_data_type(rest)
        columns.append({
            "col": col_name,
            "type_str": rest,
            "data_type": data_type,
            "not_null": not_null,
        })
    return {"table_name": table_name, "columns": columns}


# ---------------------------------------------------------------------------
# 3. Parse CREATE [MATERIALIZED] VIEW
# ---------------------------------------------------------------------------

_RE_CREATE_VIEW = re.compile(
    r'CREATE\s+(?P<mat>MATERIALIZED\s+)?VIEW\s+(?P<name>[\w."]+)\s+AS\s+'
    r'(?P<query>.+?)(?:\s+WITH\s+(?:NO\s+)?DATA\s*)?;',
    re.S | re.I,
)


def _parse_create_view(stmt: str) -> dict | None:
    m = _RE_CREATE_VIEW.search(stmt)
    if not m:
        return None
    is_mat = bool(m.group("mat"))
    view_name = m.group("name").replace('"', '')
    query = m.group("query").strip()
    query = re.sub(r'\s+WITH\s+(NO\s+)?DATA\s*$', '', query, flags=re.I)
    return {
        "table_type": "materialized_view" if is_mat else "view",
        "table_name": view_name,
        "query": query,
    }


# ---------------------------------------------------------------------------
# 4. Parse constraints
# ---------------------------------------------------------------------------

_RE_PK = re.compile(
    r'ALTER\s+TABLE\s+(?:ONLY\s+)?(?P<tbl>[\w."]+)\s+ADD\s+CONSTRAINT\s+\w+\s+'
    r'PRIMARY\s+KEY\s*\((?P<cols>[^)]+)\)', re.I | re.S)
_RE_FK = re.compile(
    r'ALTER\s+TABLE\s+(?:ONLY\s+)?(?P<tbl>[\w."]+)\s+ADD\s+CONSTRAINT\s+\w+\s+'
    r'FOREIGN\s+KEY\s*\((?P<cols>[^)]+)\)\s*REFERENCES\s+(?P<ref_tbl>[\w."]+)'
    r'\s*\((?P<ref_cols>[^)]+)\)', re.I | re.S)
_RE_UQ = re.compile(
    r'ALTER\s+TABLE\s+(?:ONLY\s+)?(?P<tbl>[\w."]+)\s+ADD\s+CONSTRAINT\s+\w+\s+'
    r'UNIQUE\s*\((?P<cols>[^)]+)\)', re.I | re.S)


def _parse_constraints(stmts):
    pks, fks, uqs = {}, {}, {}
    for s in stmts:
        m = _RE_PK.search(s)
        if m:
            tbl = m.group("tbl").replace('"', '')
            pks.setdefault(tbl, set()).update(
                c.strip().replace('"', '') for c in m.group("cols").split(","))
        m = _RE_FK.search(s)
        if m:
            tbl = m.group("tbl").replace('"', '')
            cols = [c.strip().replace('"', '') for c in m.group("cols").split(",")]
            ref_tbl = m.group("ref_tbl").replace('"', '')
            ref_cols = [c.strip().replace('"', '') for c in m.group("ref_cols").split(",")]
            d = fks.setdefault(tbl, {})
            for c, rc in zip(cols, ref_cols):
                d[c] = (ref_tbl, rc)
        m = _RE_UQ.search(s)
        if m:
            tbl = m.group("tbl").replace('"', '')
            uqs.setdefault(tbl, set()).update(
                c.strip().replace('"', '') for c in m.group("cols").split(","))
    return pks, fks, uqs


# ---------------------------------------------------------------------------
# 5. Parse CREATE INDEX
# ---------------------------------------------------------------------------

_RE_INDEX = re.compile(
    r'CREATE\s+(?P<uniq>UNIQUE\s+)?INDEX\s+(?P<idx_name>[\w."]+)\s+'
    r'ON\s+(?P<tbl>[\w."]+)\s+USING\s+(?P<algo>\w+)\s*'
    r'\((?P<cols>[^)]+)\)', re.I | re.S)


def _parse_indexes(stmts):
    idx_info, uq_from_idx = {}, {}
    for s in stmts:
        m = _RE_INDEX.search(s)
        if not m:
            continue
        tbl = m.group("tbl").replace('"', '')
        algo = m.group("algo").lower()
        is_unique = bool(m.group("uniq"))
        for part in _split_top_level(m.group("cols").strip(), ','):
            tokens = part.strip().split()
            col_name = tokens[0].replace('"', '')
            ops_name = tokens[1] if len(tokens) > 1 else pd.NA
            idx_info.setdefault(tbl, {})[col_name] = (algo, ops_name)
            if is_unique:
                uq_from_idx.setdefault(tbl, set()).add(col_name)
    return idx_info, uq_from_idx


# ---------------------------------------------------------------------------
# 6. SQL expression column-reference extraction
# ---------------------------------------------------------------------------

_SQL_KEYWORDS = {
    'select', 'from', 'where', 'group', 'by', 'order', 'having',
    'and', 'or', 'not', 'in', 'is', 'null', 'as', 'on', 'join',
    'left', 'right', 'inner', 'outer', 'cross', 'lateral', 'true',
    'false', 'case', 'when', 'then', 'else', 'end', 'distinct',
    'asc', 'desc', 'limit', 'offset', 'union', 'all', 'exists',
    'between', 'like', 'ilike', 'cast', 'over', 'partition',
    'row_number', 'rank', 'count', 'sum', 'avg', 'min', 'max',
    'array_agg', 'unnest', 'coalesce', 'nullif', 'greatest', 'least',
    'ln', 'log', 'exp', 'abs', 'ceil', 'floor', 'round',
    'text', 'integer', 'smallint', 'bigint', 'numeric', 'double',
    'precision', 'boolean', 'date', 'timestamp', 'varchar', 'char',
    'jsonb', 'json', 'with', 'filter', 'within', 'rows', 'range',
    'unbounded', 'preceding', 'following', 'current', 'row', 'no',
    'data', 'materialized', 'view', 'create', 'table', 'index',
    'using', 'if', 'only', 'into', 'values', 'set', 'update',
    'delete', 'insert', 'drop', 'alter', 'add', 'column',
}


def _extract_column_refs(expr: str) -> list[tuple[str, str]]:
    expr = _sanitize_pg_operators(expr)
    refs = []

    # 3-part: schema.table.col
    for m in re.finditer(r'(?<![.\w])(\w+)\.(\w+)\.(\w+)(?![.\w])', expr):
        refs.append((f"{m.group(1)}.{m.group(2)}", m.group(3)))

    masked = re.sub(r'(?<![.\w])\w+\.\w+\.\w+(?![.\w])', '___3M___', expr)

    # 2-part: table.col (not followed by '(' which would be schema.func)
    for m in re.finditer(r'(?<![.\w])(\w+)\.(\w+)(?![.\w(])', masked):
        tbl, col = m.group(1), m.group(2)
        if tbl.lower() in ('public', 'pg_catalog', 'information_schema'):
            continue
        if m.start() > 1 and masked[m.start()-2:m.start()] == '::':
            continue
        refs.append((tbl, col))

    if not refs:
        for m in re.finditer(r'(?<![.\w])(\w+)(?![.\w(])', masked):
            w = m.group(1)
            if w.lower() in _SQL_KEYWORDS or w.isdigit():
                continue
            if m.start() >= 2 and masked[m.start()-2:m.start()] == '::':
                continue
            refs.append(("", w))

    return refs


def _extract_func_arg_refs(func_call_sql: str) -> list[tuple[str, str]]:
    m = re.search(r'[\w.]+\s*\(', func_call_sql)
    if not m:
        return []
    start = m.end() - 1
    depth, end = 0, start
    for i in range(start, len(func_call_sql)):
        if func_call_sql[i] == '(':
            depth += 1
        elif func_call_sql[i] == ')':
            depth -= 1
        if depth == 0:
            end = i
            break
    args_str = func_call_sql[start + 1:end]
    return _extract_column_refs(args_str)


# ---------------------------------------------------------------------------
# 7. FROM clause parsing
# ---------------------------------------------------------------------------

_JOIN_RE = re.compile(
    r'\b(CROSS\s+JOIN\s+LATERAL|CROSS\s+JOIN|LEFT\s+JOIN\s+LATERAL|'
    r'LEFT\s+OUTER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|'
    r'INNER\s+JOIN|JOIN\s+LATERAL|JOIN)\b',
    re.I,
)


def _strip_outer_parens_from_item(sql: str) -> str:
    s = sql.strip()
    if not s.startswith('(') or not s.endswith(')'):
        return s
    inner = _strip_outer_parens(s)
    if re.match(r'\s*(?:WITH\b|SELECT\b)', inner, re.I):
        return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return s
    return inner


def _parse_from_clause(sql: str) -> list[dict]:
    sql = sql.strip()
    if not sql:
        return []
    m = re.match(r'\bFROM\b\s*', sql, re.I)
    if m:
        sql = sql[m.end():]
    for kw in ["WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "WINDOW",
               "UNION", "INTERSECT", "EXCEPT", "FETCH"]:
        pos = _find_keyword_pos(sql, kw)
        if pos > 0:
            sql = sql[:pos].strip()
    sources = []
    _parse_from_items(sql, sources)
    return sources


def _parse_from_items(sql: str, out: list[dict]):
    sql = sql.strip()
    if not sql:
        return
    parts = _split_top_level(sql, ',')
    for part in parts:
        part = part.strip()
        if part:
            _parse_join_chain(part, out)


def _parse_join_chain(sql: str, out: list[dict]):
    sql = sql.strip()
    joins = []
    depth, in_sq, in_dq = 0, False, False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
        if depth == 0 and not in_sq and not in_dq:
            m = _JOIN_RE.match(sql, i)
            if m:
                joins.append((m.start(), m.end(), m.group(0)))
                i = m.end()
                continue
        i += 1

    if not joins:
        _parse_single_source(sql, out)
        return

    first = sql[:joins[0][0]].strip()
    if first:
        _parse_single_source(first, out)

    for idx, (start, end, join_kw) in enumerate(joins):
        next_start = joins[idx + 1][0] if idx + 1 < len(joins) else len(sql)
        source_text = sql[end:next_start].strip()
        on_pos = _find_keyword_pos(source_text, "ON")
        using_pos = _find_keyword_pos(source_text, "USING")
        cut = len(source_text)
        if on_pos > 0:
            cut = min(cut, on_pos)
        if using_pos > 0:
            cut = min(cut, using_pos)
        source_only = source_text[:cut].strip()
        _parse_single_source(source_only, out,
                             is_lateral="LATERAL" in join_kw.upper())


def _parse_single_source(sql: str, out: list[dict], is_lateral: bool = False):
    sql = sql.strip()
    sql = _strip_outer_parens_from_item(sql)
    if not sql:
        return

    m_lat = re.match(r'\bLATERAL\b\s*', sql, re.I)
    if m_lat:
        is_lateral = True
        sql = sql[m_lat.end():].strip()

    # --- Subquery ---
    if sql.startswith('(') or re.match(r'\bSELECT\b', sql, re.I):
        if sql.startswith('('):
            depth, ci = 0, 0
            for ci in range(len(sql)):
                if sql[ci] == '(':
                    depth += 1
                elif sql[ci] == ')':
                    depth -= 1
                if depth == 0:
                    break
            after = sql[ci + 1:].strip()
            m_alias = re.match(r'(?:AS\s+)?(\w+)', after, re.I)
            alias = m_alias.group(1).lower() if m_alias else ""
            inner = sql[1:ci].strip()
        else:
            inner = sql
            alias = ""
        out.append({
            "type": "lateral_subquery" if is_lateral else "subquery",
            "name": alias,
            "alias": alias,
            "subquery_sql": inner,
            "func_call_sql": None,
        })
        return

    # --- Function call ---
    m_func = re.match(r'([\w.]+)\s*\(', sql, re.I)
    if m_func:
        func_name = m_func.group(1).lower()
        start_paren = m_func.end() - 1
        depth, ci = 0, start_paren
        for ci in range(start_paren, len(sql)):
            if sql[ci] == '(':
                depth += 1
            elif sql[ci] == ')':
                depth -= 1
            if depth == 0:
                break
        func_call_full = sql[:ci + 1]
        after = sql[ci + 1:].strip()
        alias = ""
        col_aliases = []
        m_alias = re.match(r'(?:AS\s+)?(\w+)\s*(?:\(([^)]*)\))?', after, re.I)
        if m_alias:
            alias = m_alias.group(1).lower()
            if m_alias.group(2):
                col_aliases = [c.strip().lower()
                               for c in m_alias.group(2).split(',')]
        out.append({
            "type": "lateral_func" if is_lateral else "func",
            "name": func_name,
            "alias": alias or func_name,
            "subquery_sql": None,
            "func_call_sql": func_call_full,
            "col_aliases": col_aliases,
        })
        return

    # --- Contains JOINs? ---
    if _find_keyword_pos(sql, "JOIN") >= 0:
        _parse_join_chain(sql, out)
        return

    # --- Plain table ---
    tokens = sql.split()
    tokens = [t for t in tokens if t.upper() != "AS"]
    if len(tokens) >= 2:
        table_name = tokens[0].replace('"', '').lower()
        alias = tokens[1].replace('"', '').lower()
    elif len(tokens) == 1:
        table_name = tokens[0].replace('"', '').lower()
        alias = table_name.split('.')[-1]
    else:
        return
    out.append({
        "type": "table",
        "name": table_name,
        "alias": alias,
        "subquery_sql": None,
        "func_call_sql": None,
    })


# ---------------------------------------------------------------------------
# 8. CTE extraction
# ---------------------------------------------------------------------------

def _extract_ctes(sql: str) -> tuple[dict[str, str], str]:
    sql = sql.strip()
    if not re.match(r'\bWITH\b', sql, re.I):
        return {}, sql
    rest = sql[4:].strip()
    ctes: dict[str, str] = {}
    while True:
        m = re.match(r'(\w+)\s+AS\s*\(', rest, re.I)
        if not m:
            break
        cte_name = m.group(1).lower()
        start = m.end() - 1
        depth, i = 0, start
        for i in range(start, len(rest)):
            if rest[i] == '(':
                depth += 1
            elif rest[i] == ')':
                depth -= 1
            if depth == 0:
                break
        body = rest[start + 1:i].strip()
        ctes[cte_name] = body
        rest = rest[i + 1:].strip()
        if rest.startswith(','):
            rest = rest[1:].strip()
        else:
            break
    return ctes, rest


# ---------------------------------------------------------------------------
# 9. SELECT column extraction
# ---------------------------------------------------------------------------

def _extract_select_columns(sql: str) -> tuple[list[tuple[str, str]], str]:
    sql = sql.strip()
    m = re.match(r'(?:SELECT\s+(?:DISTINCT\s+)?)', sql, re.I)
    if m:
        sql = sql[m.end():]
    from_pos = _find_keyword_pos(sql, "FROM")
    if from_pos < 0:
        col_part, rest = sql, ""
    else:
        col_part, rest = sql[:from_pos].strip(), sql[from_pos:]

    parts = _split_top_level(col_part, ',')
    columns = []
    for p in parts:
        p = p.strip()
        as_pos = -1
        depth = 0
        for i in range(len(p)):
            if p[i] == '(':
                depth += 1
            elif p[i] == ')':
                depth -= 1
            if depth == 0 and p[i:i + 4].upper() == ' AS ':
                as_pos = i + 1
                break
            if depth == 0 and p[i:i + 3].upper() == ' AS' and i + 3 == len(p):
                as_pos = i + 1
                break
        if as_pos > 0:
            expr = p[:as_pos - 1].strip()
            alias = p[as_pos + 2:].strip().strip('"').lower()
        else:
            expr = p
            m2 = re.search(r'(\w+)\s*$', p)
            alias = m2.group(1).lower() if m2 else p.strip().lower()
        columns.append((alias, expr))
    return columns, rest


# ---------------------------------------------------------------------------
# 10. Column lineage tracing
# ---------------------------------------------------------------------------

def _resolve_column_sources(query_sql: str,
                            known_tables: set[str]) -> dict[str, set[str]]:
    query_sql = _sanitize_pg_operators(query_sql)
    cte_raw, main_sql = _extract_ctes(query_sql)
    cte_lineage: dict[str, dict[str, set[str]]] = {}

    for cte_name, cte_body in cte_raw.items():
        col_map = _trace_select(cte_body, known_tables, cte_lineage, cte_raw)
        cte_lineage[cte_name] = col_map

    final_col_map = _trace_select(main_sql, known_tables,
                                  cte_lineage, cte_raw)

    result = {}
    for col_name, srcs in final_col_map.items():
        resolved = set()
        for s in srcs:
            resolved.update(_resolve_to_base(s, cte_lineage, known_tables))
        result[col_name] = resolved
    return result


def _trace_select(
    sql: str,
    known_tables: set[str],
    cte_lineage: dict[str, dict[str, set[str]]],
    cte_raw: dict[str, str],
) -> dict[str, set[str]]:
    sql = sql.strip()
    sql = _strip_outer_parens(sql)

    inner_cte_raw, sql = _extract_ctes(sql)
    if inner_cte_raw:
        for cn, cb in inner_cte_raw.items():
            col_map = _trace_select(cb, known_tables, cte_lineage, cte_raw)
            cte_lineage[cn] = col_map
        cte_raw = {**cte_raw, **inner_cte_raw}

    columns, rest = _extract_select_columns(sql)
    from_sources = _parse_from_clause(rest)

    alias_map: dict[str, str] = {}
    func_alias_arg_refs: dict[str, list[tuple[str, str]]] = {}

    for src in from_sources:
        alias = src["alias"]
        if src["type"] == "table":
            alias_map[alias] = src["name"]

        elif src["type"] in ("subquery", "lateral_subquery") \
                and src.get("subquery_sql"):
            alias_map[alias] = alias
            sub_col_map = _trace_select(
                src["subquery_sql"], known_tables, cte_lineage, cte_raw)
            cte_lineage[alias] = sub_col_map

        elif src["type"] in ("func", "lateral_func") \
                and src.get("func_call_sql"):
            arg_refs = _extract_func_arg_refs(src["func_call_sql"])
            func_alias_arg_refs[alias] = arg_refs

    col_map: dict[str, set[str]] = {}

    for col_alias, expr in columns:
        raw_refs = _extract_column_refs(expr)
        resolved: set[str] = set()

        for tbl_ref, col_ref in raw_refs:
            tbl_low = tbl_ref.lower() if tbl_ref else ""
            col_low = col_ref.lower()

            if tbl_low and tbl_low in alias_map:
                resolved.add(f"{alias_map[tbl_low]}.{col_low}")

            elif tbl_low and tbl_low in func_alias_arg_refs:
                _resolve_func_refs(tbl_low, func_alias_arg_refs,
                                   alias_map, resolved)

            elif tbl_low:
                resolved.add(f"{tbl_low}.{col_low}")

            else:
                found = False
                for a, rt in alias_map.items():
                    norm = _normalize_table_name(rt)
                    if norm in cte_lineage and col_low in cte_lineage[norm]:
                        resolved.add(f"{rt}.{col_low}")
                        found = True
                        break
                if not found:
                    if len(alias_map) == 1:
                        rt = list(alias_map.values())[0]
                        resolved.add(f"{rt}.{col_low}")
                    else:
                        resolved.add(col_low)

        col_map[col_alias] = resolved

    return col_map


def _resolve_func_refs(func_alias: str,
                       func_alias_arg_refs: dict[str, list[tuple[str, str]]],
                       alias_map: dict[str, str],
                       out: set[str],
                       visited: set[str] | None = None):
    """
    Recursively resolve function-alias references to real table columns.
    Handles chained function calls (func output fed into another func).
    """
    if visited is None:
        visited = set()
    if func_alias in visited:
        return
    visited = visited | {func_alias}

    for arg_tbl, arg_col in func_alias_arg_refs.get(func_alias, []):
        arg_tbl_low = arg_tbl.lower() if arg_tbl else ""
        arg_col_low = arg_col.lower()

        if arg_tbl_low and arg_tbl_low in alias_map:
            out.add(f"{alias_map[arg_tbl_low]}.{arg_col_low}")
        elif arg_tbl_low and arg_tbl_low in func_alias_arg_refs:
            _resolve_func_refs(arg_tbl_low, func_alias_arg_refs,
                               alias_map, out, visited)
        elif arg_tbl_low:
            out.add(f"{arg_tbl_low}.{arg_col_low}")
        else:
            out.add(arg_col_low)


def _resolve_to_base(
    source: str,
    cte_lineage: dict[str, dict[str, set[str]]],
    known_tables: set[str],
    visited: set[str] | None = None,
) -> set[str]:
    if visited is None:
        visited = set()
    if source in visited:
        return {source}
    visited = visited | {source}

    if "." not in source:
        return {source}

    parts = source.split(".")
    if len(parts) == 3:
        schema, table, col = parts
        full_table = f"{schema}.{table}"
        norm = _normalize_table_name(full_table)
    elif len(parts) == 2:
        table, col = parts
        full_table = table
        norm = _normalize_table_name(table)
    else:
        return {source}

    if norm in known_tables:
        canonical = f"public.{norm}" if "." not in full_table else full_table
        return {f"{canonical}.{col}"}

    if norm in cte_lineage:
        cte_col_map = cte_lineage[norm]
        if col in cte_col_map:
            resolved = set()
            for sub in cte_col_map[col]:
                resolved.update(_resolve_to_base(sub, cte_lineage,
                                                 known_tables, visited))
            return resolved

    return {source}


# ---------------------------------------------------------------------------
# 11. Fallback view column extraction
# ---------------------------------------------------------------------------

def _extract_view_columns_regex(query: str) -> list[str]:
    m = re.search(r'\bSELECT\s+(.*?)\s+FROM\b', query, re.S | re.I)
    if not m:
        return []
    parts = _split_top_level(m.group(1), ',')
    cols = []
    for p in parts:
        p = p.strip()
        am = re.search(r'\bAS\s+(\w+)\s*$', p, re.I)
        if am:
            cols.append(am.group(1).lower())
        else:
            tokens = p.split()
            cols.append(tokens[-1].strip('"').lower())
    return cols


# ---------------------------------------------------------------------------
# 12. Main orchestrator
# ---------------------------------------------------------------------------

def parse_pg_dump(path: str) -> pd.DataFrame:
    raw = _read_sql(path)
    stmts = _split_statements(raw)

    tables_info, views_info = [], []
    known_table_names: set[str] = set()

    for s in stmts:
        ti = _parse_create_table(s)
        if ti:
            tables_info.append(ti)
            known_table_names.add(_normalize_table_name(ti["table_name"]))
            continue
        vi = _parse_create_view(s)
        if vi:
            views_info.append(vi)

    pks, fks, uqs = _parse_constraints(stmts)
    idx_info, uq_from_idx = _parse_indexes(stmts)
    for tbl, cols in uq_from_idx.items():
        uqs.setdefault(tbl, set()).update(cols)

    rows = []

    for ti in tables_info:
        tbl = ti["table_name"]
        pk_cols = pks.get(tbl, set())
        fk_map = fks.get(tbl, {})
        uq_cols = uqs.get(tbl, set())
        ix = idx_info.get(tbl, {})
        for c in ti["columns"]:
            col = c["col"]
            is_pk = col in pk_cols
            is_fk = col in fk_map
            is_uq = (col in uq_cols) and not is_pk
            algo, ops = ix.get(col, (pd.NA, pd.NA))
            rows.append({
                "table_type": "table",
                "table_name": tbl,
                "column_name": col,
                "data_type": c["data_type"],
                "column_src": pd.NA,
                "is_pk": is_pk,
                "is_fk": is_fk,
                "fk_table": fk_map[col][0] if is_fk else pd.NA,
                "fk_col": fk_map[col][1] if is_fk else pd.NA,
                "is_uq": is_uq,
                "index_algo": algo,
                "index_name": ops,
                "required": c["not_null"] or is_pk,
            })

    for vi in views_info:
        tbl = vi["table_name"]
        ttype = vi["table_type"]
        query = vi["query"]
        col_sources = _resolve_column_sources(query, known_table_names)
        if not col_sources:
            sanitized = _sanitize_pg_operators(query)
            for cn in _extract_view_columns_regex(sanitized):
                col_sources.setdefault(cn, set())
        ix = idx_info.get(tbl, {})
        for col_name, srcs in col_sources.items():
            src_str = ", ".join(sorted(srcs)) if srcs else pd.NA
            algo, ops = ix.get(col_name, (pd.NA, pd.NA))
            rows.append({
                "table_type": ttype,
                "table_name": tbl,
                "column_name": col_name,
                "data_type": pd.NA,
                "column_src": src_str,
                "is_pk": False,
                "is_fk": False,
                "fk_table": pd.NA,
                "fk_col": pd.NA,
                "is_uq": False,
                "index_algo": algo,
                "index_name": ops,
                "required": False,
            })

    return pd.DataFrame(rows, columns=[
        "table_type", "table_name", "column_name", "data_type",
        "column_src",
        "is_pk", "is_fk", "fk_table", "fk_col", "is_uq",
        "index_algo", "index_name", "required",
    ])

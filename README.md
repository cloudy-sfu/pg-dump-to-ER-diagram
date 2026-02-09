# "pg_dump" to ER diagram
Convert "pg_dump" result (PostgreSQL schema) to entity-relationship diagram

![](https://shields.io/badge/dependencies-Python_3.13-blue)

## Install

Create and activate Python 3.13 virtual environment.

Run the following command in terminal.

```bash
pip install -r requirements.txt
```

Confirm "pg_dump" is installed. Source: http://enterprisedb.com/download-postgresql-binaries



## Usage

Let the connection string of PostgreSQL be `$connection_string`.

Let the installation path of "pg_dump" be `$pg_dump_path` (the folder path which contains `pg_dump.exe`).

Activate Python virtual environment.

Run the following command.

```bash
"$pg_dump_path/pg_dump" $connection_string --schema-only --no-owner --no-privileges --no-tablespaces > "raw/database_schema.sql"
python main.py "raw/database_schema.sql"
```

The program outputs ER diagrams:

-   In format of mermaid code at `raw/er_diagram.md` 
-   In format of HTML preview at `raw/er_diagram.html` (adaptive to browser's light & dark theme; support zooming and padding)

>   [!NOTE]
>
>   Allow customized input path of "pg_dump" extracted PostgreSQL schema.
>
>   The output path will be in the same folder as input path.


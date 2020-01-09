import psycopg2 as pg
import yaml
from pathlib import Path
import os
import pandas as pd

def create_tables(config, connection):
    cur = connection.cursor()
    for table in config:
        name = table.get('name')
        schema = table.get('schema')
        ddl = f"""CREATE TABLE IF NOT EXISTS {name} ({schema})"""
        cur.execute(ddl)

    connection.commit()

def transform_tables(config):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = dir_path + "/data/"

    for table in config:
        table_name = table.get('name')
        table_source = os.path.join(data_path, f"{table_name}.csv") #data_path.joinpath(f"{table_name}.csv")
        table_cols = []
        for i in table.get('columns'):
            table_cols.append(str.upper(i))
        df = pd.read_csv(table_source)
        df_reorder = df[table_cols]  # rearrange column here
        df_reorder.to_csv(table_source, index=False)

def load_tables(config, connection):

    # iterate and load
    cur = connection.cursor()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = dir_path + "/data/"

    for table in config:
        table_name = table.get('name')
        table_source = os.path.join(data_path, f"{table_name}.csv")
        with open(table_source, 'r') as f:
            next(f)
            cur.copy_expert(f"COPY {table_name} FROM STDIN CSV NULL AS ''", f)
        connection.commit()


connection = pg.connect(
    host='localhost',
    port=54320,
    dbname='trans_db',
    user='postgres'
)

with open("schemas.yaml") as schema_file:
    config = yaml.load(schema_file)
create_tables(config, connection)
transform_tables(config)
load_tables(config, connection)

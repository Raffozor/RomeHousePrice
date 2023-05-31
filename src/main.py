import pandas as pd
from sqlalchemy import create_engine
from SQL_CONFIG import SQL_CONNECTION, TABLE_NAME

# Crea la connessione al database
engine = create_engine(SQL_CONNECTION)
print(engine)
# Nome della tabella da caricare
table_name = TABLE_NAME

# Carica la tabella in un DataFrame
data = pd.read_sql_table(table_name, con=engine)

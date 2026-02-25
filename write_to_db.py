import pandas as pd

df.to_sql("sap_tcodes", engine, if_exists="replace", index=False)

print("SAP T-codes loaded successfully")
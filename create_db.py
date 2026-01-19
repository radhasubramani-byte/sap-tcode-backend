# Create SQLite DB
import pandas as pd
from sqlalchemy import create_engine   
engine = create_engine("sqlite:///sap_tcodes.db")
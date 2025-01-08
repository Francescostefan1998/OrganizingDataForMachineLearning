import pandas as pd
from io import StringIO
csv_data  = \
csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))
df



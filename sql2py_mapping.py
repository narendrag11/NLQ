from datetime import timedelta
from sqlalchemy import (
    INTEGER, BIGINT, SMALLINT, DECIMAL, NUMERIC, FLOAT, DOUBLE, CHAR, NVARCHAR, TEXT, DATE, DATETIME, TIMESTAMP, TIME, BOOLEAN, BLOB, JSON, ARRAY, UUID,Enum,Interval
)
def map_sqlalchemy_type(sqlalchemy_type):
    """
    Maps SQLAlchemy column types to their equivalent Python data types, handling multiple databases.
    """
    if hasattr(sqlalchemy_type, 'python_type'):
        return sqlalchemy_type.python_type
    
    # Handling types that don't have a direct python_type mapping
    elif str(sqlalchemy_type).startswith("ARRAY"):
        return list  # Represent ARRAY as list
    
    elif str(sqlalchemy_type).startswith("STRUCT"):
        return dict  # Represent STRUCT as dictionary
    
    elif str(sqlalchemy_type).startswith("GEOGRAPHY"):
        return "GEOGRAPHY"  # Custom string representation
    
    elif str(sqlalchemy_type).startswith("JSON"):
        return dict  # JSON should be mapped as a dictionary
    
    elif str(sqlalchemy_type).startswith("INTERVAL"):
        return timedelta  # Represent INTERVAL as a timedelta equivalent
    
    elif str(sqlalchemy_type).startswith("NUMERIC") or str(sqlalchemy_type).startswith("BIGNUMERIC"):
        return float  # BigQuery's NUMERIC can be mapped to float or Decimal
    
    elif str(sqlalchemy_type).startswith("RANGE"):
        return "RANGE"  # Custom representation for range types (PostgreSQL)

    # MySQL Specific Types
    elif str(sqlalchemy_type).startswith("TINYINT"):
        return bool  # MySQL's TINYINT(1) is usually used as Boolean
    
    elif str(sqlalchemy_type).startswith("YEAR"):
        return int  # MySQL's YEAR type maps to an integer
    
    # Oracle Specific Types
    elif str(sqlalchemy_type).startswith("RAW"):
        return bytes  # Oracle RAW maps to bytes
    
    elif str(sqlalchemy_type).startswith("CLOB") or str(sqlalchemy_type).startswith("TEXT"):
        return str  # CLOB and TEXT should be treated as strings
    
    elif str(sqlalchemy_type).startswith("BLOB"):
        return bytes  # Oracle and MySQL BLOB types map to binary
    
    # PostgreSQL Specific Types
    elif str(sqlalchemy_type).startswith("UUID"):
        return str  # PostgreSQL UUID type can be stored as a string
    
    elif str(sqlalchemy_type).startswith("INET") or str(sqlalchemy_type).startswith("CIDR"):
        return str  # IP address types should be stored as strings
    
    elif str(sqlalchemy_type).startswith("TSVECTOR"):
        return str  # PostgreSQL full-text search type
    
    # Fallback for unknown types
    else:
        return str(sqlalchemy_type)  # Return as a string for unsupported types

columns=[
    {'name': 'employee_id', 'type': INTEGER(), 'nullable': False, 'default': None, 'primary_key': 1},
    {'name': 'transaction_id', 'type': BIGINT(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'department_id', 'type': SMALLINT(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'status_code', 'type': "TINYINT()", 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'salary', 'type': DECIMAL(10, 2), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'tax_rate', 'type': NUMERIC(10, 2), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'discount_percentage', 'type': FLOAT(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'price', 'type': DOUBLE(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'country_code', 'type': CHAR(10), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'customer_name', 'type': NVARCHAR(length=255), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'product_description', 'type': TEXT(), 'nullable': True, 'default': None, 'primary_key': 0},
    {'name': 'birth_date', 'type': DATE(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'order_timestamp', 'type': DATETIME(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'last_updated', 'type': TIMESTAMP(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'shift_start_time', 'type': TIME(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'graduation_year', 'type': INTEGER(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'is_active', 'type': BOOLEAN(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'employee_photo', 'type': BLOB(), 'nullable': True, 'default': None, 'primary_key': 0},
    {'name': 'metadata', 'type': JSON(), 'nullable': True, 'default': None, 'primary_key': 0},
    {'name': 'scores', 'type': ARRAY(INTEGER()), 'nullable': True, 'default': None, 'primary_key': 0},
    {'name': 'customer_details', 'type': JSON(), 'nullable': True, 'default': None, 'primary_key': 0},  # STRUCT mapped to JSON
    {'name': 'user_uuid', 'type': UUID(), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'duration', 'type': Interval(), 'nullable': True, 'default': None, 'primary_key': 0},
    {'name': 'account_status', 'type': Enum('active', 'inactive'), 'nullable': False, 'default': None, 'primary_key': 0},
    {'name': 'color_preferences', 'type': JSON(), 'nullable': True, 'default': None, 'primary_key': 0},  # SET mapped to JSON
    {'name': 'long_description', 'type': TEXT(), 'nullable': True, 'default': None, 'primary_key': 0},  # CLOB mapped to TEXT
    {'name': 'multilingual_description', 'type': TEXT(), 'nullable': True, 'default': None, 'primary_key': 0},  # NCLOB mapped to TEXT
    {'name': 'encryption_key', 'type': BLOB(), 'nullable': True, 'default': None, 'primary_key': 0},  # RAW mapped to BLOB
    {'name': 'row_identifier', 'type': TEXT(), 'nullable': False, 'default': None, 'primary_key': 0},  # ROWID mapped to TEXT
    {'name': 'location', 'type': JSON(), 'nullable': True, 'default': None, 'primary_key': 0},  # GEOGRAPHY mapped to JSON
    {'name': 'geo_shape', 'type': JSON(), 'nullable': True, 'default': None, 'primary_key': 0}  # GEOMETRY mapped to JSON
]

# Usage Example:
type_mapping = {}
for column in columns:
    col_name = column['name']
    sqlalchemy_type = column['type']
    python_type = map_sqlalchemy_type(sqlalchemy_type)
    type_mapping[col_name] = python_type
    

for k,v in type_mapping.items():
    print("column name:    ",k,"    and column type      ",v)




exp="""

More precise handling of unsupported types: Added elif cases for various databases.
Custom mapping for unsupported SQLAlchemy types:
ARRAY → list
STRUCT → dict
GEOGRAPHY → "GEOGRAPHY"
JSON → dict
INTERVAL → "timedelta"
NUMERIC/BIGNUMERIC → float
RANGE → "RANGE"
MySQL-specific mappings:
TINYINT(1) → bool
YEAR → int
Oracle-specific mappings:
RAW → bytes
CLOB/TEXT → str
BLOB → bytes
PostgreSQL-specific mappings:
UUID → str
INET/CIDR → str
TSVECTOR → str
Fallback to str(sqlalchemy_type) for unknown types.

"""

ouput=["""
column name:     employee_id     and column type       <class 'int'>
column name:     transaction_id     and column type       <class 'int'>
column name:     department_id     and column type       <class 'int'>
column name:     status_code     and column type       <class 'bool'>
column name:     salary     and column type       <class 'decimal.Decimal'>
column name:     tax_rate     and column type       <class 'decimal.Decimal'>
column name:     discount_percentage     and column type       <class 'float'>
column name:     price     and column type       <class 'float'>
column name:     country_code     and column type       <class 'str'>
column name:     customer_name     and column type       <class 'str'>
column name:     product_description     and column type       <class 'str'>
column name:     birth_date     and column type       <class 'datetime.date'>
column name:     order_timestamp     and column type       <class 'datetime.datetime'>
column name:     last_updated     and column type       <class 'datetime.datetime'>
column name:     shift_start_time     and column type       <class 'datetime.time'>
column name:     graduation_year     and column type       <class 'int'>
column name:     is_active     and column type       <class 'bool'>
column name:     employee_photo     and column type       <class 'bytes'>
column name:     metadata     and column type       <class 'dict'>
column name:     scores     and column type       <class 'list'>
column name:     customer_details     and column type       <class 'dict'>
column name:     user_uuid     and column type       <class 'uuid.UUID'>
column name:     duration     and column type       <class 'datetime.timedelta'>
column name:     account_status     and column type       <class 'str'>
column name:     color_preferences     and column type       <class 'dict'>
column name:     long_description     and column type       <class 'str'>
column name:     multilingual_description     and column type       <class 'str'>
column name:     encryption_key     and column type       <class 'bytes'>
column name:     row_identifier     and column type       <class 'str'>
column name:     location     and column type       <class 'dict'>
column name:     geo_shape     and column type       <class 'dict'>

"""]

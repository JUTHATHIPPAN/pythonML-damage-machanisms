import pymssql

server = '52.220.173.180'
user = 'sa'
password = 'A!dex0n7845'
database = 'ML_damage'
port = 1433  # Replace with your custom port number if applicable

conn = pymssql.connect(server=server, user=user, password=password, database=database, port=port)
cursor = conn.cursor()

# JWT Configuration
JWT_SECRET_KEY = "Predict_Damage_Mechanisms"
JWT_ALGORITHM = "HS256"
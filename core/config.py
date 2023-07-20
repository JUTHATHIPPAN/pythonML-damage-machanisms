import pyodbc

server = '52.220.173.180,1433'
username = 'sa'
password = 'A!dex0n7845'
database = 'ML_damage'
port = 1433  # Replace with your custom port number if applicable

conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password + ';TrustServerCertificate=yes;' )
cursor = conn.cursor()

# JWT Configuration
JWT_SECRET_KEY = "Predict_Damage_Mechanisms"
JWT_ALGORITHM = "HS256"
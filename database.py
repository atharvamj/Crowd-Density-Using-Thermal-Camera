import sqlite3

conn = sqlite3.connect('Database_3.db')
print(conn.total_changes)
print("Opened database successfully")
cursor = conn.cursor()
# cursor.execute("CREATE TABLE fish (name TEXT, species TEXT, tank_number INTEGER)")
# cursor.execute("INSERT INTO fish VALUES ('Sammy', 'shark', 1)")
# cursor.execute("INSERT INTO fish VALUES ('Jamie', 'cuttlefish', 7)")
print(1)
cursor.execute("CREATE TABLE People_number ( id INTEGER ,Number INTEGER)")
# cursor.execute("INSERT INTO People_number VALUES (1)")

print("Table created successfully")
conn.commit()
conn.close()
import sqlite3

conn = sqlite3.connect('Database_3.db')
print("Opened database successfully")
cursor = conn.cursor()
# cursor.execute("CREATE TABLE People_number ( Number INTEGER)")
# rows = cursor.execute("SELECT Number FROM People_number").fetchall()
rows = cursor.execute("SELECT id , Number FROM People_number").fetchall()
print(1)
for x in rows:
    print(x[0],x[1])

cursor.execute("DELETE FROM People_number WHERE number = 1",)

print("Table created successfully")
conn.commit()
conn.close()
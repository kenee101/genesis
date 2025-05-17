import sqlite3

# Connect to SQLite database
def connect_to_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    return conn, cursor

# Create a table for storing data
def create_table(cursor):
    table_info ="""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            grade TEXT NOT NULL,
            address TEXT NOT NULL,
            level TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    cursor.execute(table_info)
    cursor.connection.commit()

# Insert data into the database
def insert_data(cursor, data):
    cursor.executemany('''
        INSERT INTO students (name, age, grade, address, level)
        VALUES (?, ?, ?, ?, ?)
    ''', (data))
    cursor.connection.commit()

# Retrieve chat history from the database
def retrieve_data(cursor):
    cursor.execute('''
        SELECT * FROM students
        ORDER BY timestamp DESC
    ''')
    return cursor.fetchall()

def clean_sql_query(query):
    query = query.strip()
    if query.startswith("```sql"):
        query = query[6:]  # remove leading ```sql
    elif query.startswith("```"):
        query = query[3:]  # remove leading ```
    if query.endswith("```"):
        query = query[:-3]  # remove trailing ```
    return query.strip()

def run_agent_query(query, db_path):
    query = clean_sql_query(query)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(query)
    result = cursor.fetchall()
    return result

# Close the database connection
def close_sqlite_connection(conn):
    conn.close()

# Usage
if __name__ == "__main__":
    # Connect to the database
    conn, cursor = connect_to_db('students.db')
    
    # Create table
    create_table(cursor)
    
    # Insert data
    data = [
        ('elijah', 18, "A", 'lagos', '5'),
        ('Sophia', 20, 'B', 'Abuja', '4'),
        ('Michael', 22, 'C', 'Kano', '3'),
        ('Sarah', 19, 'A', 'Port Harcourt', '2'),
        ('David', 21, 'B', 'Ibadan', '1'),
        ('Emily', 23, 'C', 'Enugu', '5'),
        ('James', 24, 'A', 'Benin', '4'),
        ('Olivia', 25, 'B', 'Kaduna', '3'),
        ('William', 26, 'C', 'Jos', '2'),
        ('Ava', 27, 'A', 'Owerri', '1')
        ]
    insert_data(cursor, data)
    
    # Retrieve and print data
    history = retrieve_data(cursor)
    for row in history:
        print(row)
    
    # Close the database connection
    close_sqlite_connection(conn)

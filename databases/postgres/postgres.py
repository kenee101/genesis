import psycopg2

postgres_conn = psycopg2.connect(
    dbname="genesis",
    user="postgres",
    password="secret",
    host="localhost",
    port="5432"
)

cursor = postgres_conn.cursor()
# cursor.execute("SELECT * FROM genai_memory;")
# print(cursor.fetchall())

def store_prompt_data(id, prompt, response, embedding, model_used):
    cursor.execute(
        """
        INSERT INTO genai_memory (user_id, prompt, response, vector, model_used)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (id, prompt, response, embedding, model_used)
    )
    postgres_conn.commit()
    # close_postgres_connection(postgres_conn)

def store_employee_data(data):
    cursor.executemany(
        """
        INSERT INTO employees (name, age, department, salary)
        VALUES (%s, %s, %s, %s)
        """,
        (data)
    )
    postgres_conn.commit()

# Close the database connection
def close_postgres_connection(conn):
    conn.close()
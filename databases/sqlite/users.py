import sqlite3
import hashlib
import os
from typing import Optional, Tuple


def init_users_db():
    """Initialize the users database and create the users table if it doesn't exist."""
    conn = sqlite3.connect('databases/sqlite/users.db')
    cursor = conn.cursor()

    # Create users table with secure password storage
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash a password with a salt."""
    if salt is None:
        salt = os.urandom(32).hex()

    # Combine password and salt
    salted = password + salt
    # Create hash
    hash_obj = hashlib.sha256(salted.encode())
    return hash_obj.hexdigest(), salt


def register_user(username: str, password: str) -> bool:
    """Register a new user with a hashed password."""
    try:
        conn = sqlite3.connect('databases/sqlite/users.db')
        cursor = conn.cursor()

        # Hash the password
        password_hash, salt = hash_password(password)

        # Insert the new user
        cursor.execute(
            'INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)',
            (username, password_hash, salt)
        )

        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False
    except Exception as e:
        print(f"Error registering user: {e}")
        return False


def verify_user(username: str, password: str) -> bool:
    """Verify a user's credentials."""
    try:
        conn = sqlite3.connect('databases/sqlite/users.db')
        cursor = conn.cursor()

        # Get the user's salt and hash
        cursor.execute(
            'SELECT password_hash, salt FROM users WHERE username = ?',
            (username,)
        )
        result = cursor.fetchone()

        if result is None:
            return False

        stored_hash, salt = result

        # Hash the provided password with the stored salt
        password_hash, _ = hash_password(password, salt)

        conn.close()
        return password_hash == stored_hash
    except Exception as e:
        print(f"Error verifying user: {e}")
        return False


def get_all_users():
    """Get all usernames (for admin purposes)."""
    try:
        conn = sqlite3.connect('databases/sqlite/users.db')
        cursor = conn.cursor()

        cursor.execute('SELECT username, created_at FROM users')
        users = cursor.fetchall()

        conn.close()
        return users
    except Exception as e:
        print(f"Error getting users: {e}")
        return []


# Initialize the database when the module is imported
init_users_db()

-- Active: 1744505784299@@127.0.0.1@5432@genesis

-- This SQL script creates a table for storing GenAI memory in a PostgreSQL database.

CREATE TABLE genai_memory (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    prompt TEXT,
    response TEXT,
    model_used VARCHAR(100),
    vector FLOAT8[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    department VARCHAR(100),
    salary DECIMAL(10, 2)
);

ALTER TABLE employees ADD COLUMN joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

CREATE INDEX idx_user_id ON genai_memory (user_id);


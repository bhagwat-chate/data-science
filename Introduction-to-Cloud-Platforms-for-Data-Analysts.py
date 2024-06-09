# Section 1: SQL Code Snippets

# Select data from a table
# This query retrieves the name, age, and salary of employees in the Sales department.
"""
SELECT name, age, salary
FROM employees
WHERE department = 'Sales';
"""

# Aggregate data
# This query counts the number of employees and calculates the average salary per department.
"""
SELECT department, COUNT(*) AS employee_count, AVG(salary) AS avg_salary
FROM employees
GROUP BY department;
"""

# Join tables
# This query combines data from employees and departments tables to get employee names, their departments, and managers.
"""
SELECT e.name, e.department, d.manager
FROM employees e
JOIN departments d ON e.department = d.name;
"""

# Practical SQL Example: Querying a Relational Database
# These queries demonstrate basic data retrieval, joining tables, and aggregating data.
"""
-- Select data from the "employees" table
SELECT name, department, salary
FROM employees
WHERE salary > 50000;

-- Join "employees" and "departments" tables
SELECT e.name, e.salary, d.department_name
FROM employees e
JOIN departments d ON e.department_id = d.id;

-- Aggregate data with GROUP BY
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department;
"""

# Section 2: NoSQL Code Snippets (MongoDB)

# Python code to interact with MongoDB
from pymongo import MongoClient

# Connect to MongoDB
# This connects to a local MongoDB instance running on the default port.
client = MongoClient('localhost', 27017)
db = client['company_db']

# Insert a document
# This command inserts a new document into the employees collection.
db.employees.insert_one({"name": "Jane Doe", "age": 28, "department": "HR", "salary": 60000})

# Find documents
# This command retrieves all employees in the HR department.
employees = db.employees.find({"department": "HR"})
for emp in employees:
    print(emp)

# Update a document
# This command updates the salary of Jane Doe.
db.employees.update_one({"name": "Jane Doe"}, {"$set": {"salary": 65000}})

# Delete a document
# This command deletes the document for Jane Doe.
db.employees.delete_one({"name": "Jane Doe"})

# Practical NoSQL Example: Querying a MongoDB Database
# These commands show how to insert, find, update, and delete documents in MongoDB.
"""
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['company_db']

# Insert a document
db.employees.insert_one({"name": "Jane Doe", "age": 28, "department": "HR", "salary": 60000})

# Find documents
employees = db.employees.find({"department": "HR"})
for emp in employees:
    print(emp)

# Update a document
db.employees.update_one({"name": "Jane Doe"}, {"$set": {"salary": 65000}})

# Delete a document
db.employees.delete_one({"name": "Jane Doe"})
"""

# Section 3: AWS Code Snippets

# Uploading a File to S3
# This example demonstrates how to upload a file to Amazon S3.
import boto3

# Initialize a session using Amazon S3
s3 = boto3.client('s3')

# Upload a new file
s3.upload_file('local_file.txt', 'bucket_name', 'file_in_s3.txt')

# Connecting to a PostgreSQL Database on RDS
# This example demonstrates how to connect to a PostgreSQL database on Amazon RDS and run a query.
import psycopg2

# Connect to your PostgreSQL database on Amazon RDS
conn = psycopg2.connect(
    dbname="your_dbname",
    user="your_username",
    password="your_password",
    host="your_rds_endpoint",
    port="5432"
)

# Create a cursor object
cur = conn.cursor()

# Execute a SQL query
cur.execute("SELECT * FROM your_table;")

# Fetch and print the results
rows = cur.fetchall()
for row in rows:
    print(row)

# Close the cursor and connection
cur.close()
conn.close()

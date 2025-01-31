
import sqlite3

conn = sqlite3.connect("company.db")


cursor = conn.cursor()


cursor.execute('''
    CREATE TABLE IF NOT EXISTS Employees (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL,
        Department TEXT NOT NULL,
        Salary INTEGER NOT NULL,
        Hire_Date TEXT NOT NULL
    )
''')


cursor.execute('''
    CREATE TABLE IF NOT EXISTS Departments (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL UNIQUE,
        Manager TEXT NOT NULL
    )
''')


employees_data = [
    ("Alice", "Sales", 50000, "2021-01-15"),
    ("Bob", "Engineering", 70000, "2020-06-10"),
    ("Charlie", "Marketing", 60000, "2022-03-20"),
    ("David", "HR", 55000, "2019-09-01"),
    ("Emma", "Finance", 65000, "2021-05-25"),
    ("Frank", "Sales", 52000, "2022-07-10"),
    ("Grace", "Engineering", 72000, "2020-08-30"),
    ("Hannah", "Marketing", 59000, "2023-01-18"),
    ("Ian", "HR", 58000, "2018-11-12"),
    ("Jack", "Finance", 67000, "2019-04-05"),
    ("Karen", "Sales", 53000, "2022-09-15"),
    ("Liam", "Engineering", 75000, "2017-06-28"),
    ("Mia", "Marketing", 61000, "2023-05-08"),
    ("Noah", "HR", 57000, "2021-12-14"),
    ("Olivia", "Finance", 69000, "2020-03-20")
]

cursor.executemany('''
    INSERT INTO Employees (Name, Department, Salary, Hire_Date)
    VALUES (?, ?, ?, ?)
''', employees_data)


departments_data = [
    ("Sales", "Alice"),
    ("Engineering", "Bob"),
    ("Marketing", "Charlie"),
    ("HR", "David"),
    ("Finance", "Emma"),
    ("Customer Support", "Frank"),
    ("IT", "Grace"),
    ("Operations", "Hannah"),
    ("Legal", "Ian"),
    ("Procurement", "Jack")
]

cursor.executemany('''
    INSERT INTO Departments (Name, Manager)
    VALUES (?, ?)
''', departments_data)


conn.commit()
conn.close()

print("Database and tables created successfully with additional data!")

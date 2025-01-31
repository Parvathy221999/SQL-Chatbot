# SQL Query Assistant

## Overview
SQL Query Assistant is an intelligent chat-based tool designed to interpret natural language queries, convert them into optimized SQL queries, and fetch relevant data from an SQLite database. This project utilizes **Llama LLM** and is built using the **Streamlit** framework for an interactive user experience. 🤖💬

## Features
- **Natural Language Processing**: Converts user queries into SQL commands. 🔄
- **Data Interpretation**: Analyzes structured table data. 📊
- **Efficient Query Execution**: Runs optimized SQL queries in an SQLite database. ⚡
- **Formatted Results**: Presents data in a structured and readable format. 📝
- **Detailed Explanations**: Provides insights into generated SQL queries and their execution results. 🧑‍💻

## Technologies Used
- **Llama LLM**: For natural language understanding and query conversion. 🦙
- **Streamlit**: For building an interactive and user-friendly web interface. 🌐
- **SQLite**: Lightweight database to store and retrieve data efficiently. 🗃️
- **Python**: Core programming language for data processing and logic. 🐍

## Guide:
- Code run before install VS code (C++),Cmake

## Installation
To set up and run the project locally, follow these steps:

```bash
# Clone the repository
mkdir sql-query-assistant
git clone https://github.com/Parvathy221999/SQL-Chatbot.git
cd sql-query-assistant
cd Model
!wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true
cd ..
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Usage
1. Start the application using Streamlit.
2. Enter natural language queries related to the provided database tables.
3. View the generated SQL query and retrieved results.
4. Read explanations for a better understanding of SQL query logic.

## Contribution
Feel free to contribute to this project by following these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License.

## Contact
For any questions or suggestions, feel free to reach out via [Parvathyj415@gmail.com](mailto:Parvathyj415@gmail.com).


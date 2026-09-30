# Bangladesh Air Quality Monitoring Dashboard

A Streamlit-based Bangladesh Air Quality Monitoring and Machine Learning Dashboard.

## Features

- User registration
- User login
- Secure password hashing
- MySQL database backend
- Logout functionality
- Protected dashboard
- Interactive pollutant selection
- Year selection
- Monthly air-quality charts
- Monthly data table
- Random Forest temperature prediction
- Random Forest humidity prediction
- Pollutant feature importance
- Custom CSS interface
- Responsive Streamlit layout

## Dataset

The application uses:

air_quality_data.csv

The CSV should contain these columns:

- year
- month
- CO_mean
- NO2_mean
- SO2_mean
- O3_mean
- Temp_mean
- Humidity_mean

Place the CSV file in the same directory as app.py.

## Project Structure

Bangladesh_Air_Quality_Dashboard/

    app.py
    air_quality_data.csv
    requirements.txt
    .env
    .gitignore
    README.md

    database/
        database.sql

    css/
        style.css

## 1. Install Python

Python 3.10 or newer is recommended.

## 2. Create a virtual environment

Windows:

    python -m venv venv

Activate it:

    venv\Scripts\activate

## 3. Install dependencies

    pip install -r requirements.txt

## 4. Configure MySQL

Open MySQL Workbench or MySQL command line.

Run:

    database/database.sql

This creates:

    air_quality_dashboard

and:

    users

## 5. Configure .env

Open .env and change:

    MYSQL_HOST=localhost
    MYSQL_PORT=3306
    MYSQL_USER=root
    MYSQL_PASSWORD=YOUR_MYSQL_PASSWORD
    MYSQL_DATABASE=air_quality_dashboard

Replace YOUR_MYSQL_PASSWORD with your MySQL password.

## 6. Start the application

Run:

    streamlit run app.py

The application should open in your browser.

## Login

New users should first select:

    Create Account

After registering, they can log in using their username/email and password.

## Security

Passwords are not stored as plain text.

The application creates a salted PBKDF2-SHA256 password hash before storing the password in MySQL.

Never upload .env to GitHub or share your MySQL password.

## Troubleshooting

### MySQL connection error

Check that MySQL Server is running.

Also verify:

- MYSQL_HOST
- MYSQL_PORT
- MYSQL_USER
- MYSQL_PASSWORD
- MYSQL_DATABASE

in .env.

### CSV file not found

Make sure:

    air_quality_data.csv

is located next to:

    app.py

### NaN error

The application removes rows containing missing values from the columns required by each machine-learning model before training.

### Streamlit command not found

Try:

    python -m streamlit run app.py
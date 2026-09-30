CREATE DATABASE IF NOT EXISTS air_quality_dashboard;

USE air_quality_dashboard;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,

    username VARCHAR(50) NOT NULL UNIQUE,

    email VARCHAR(150) NOT NULL UNIQUE,

    password_hash VARCHAR(300) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
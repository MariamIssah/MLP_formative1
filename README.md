# Climate Data Management and Prediction System

A comprehensive machine learning pipeline project for managing and analyzing climate data, with prediction capabilities. This system provides dual API endpoints (MySQL and MongoDB) for data operations and includes ML-powered yield predictions.

## 📋 Project Overview

This project is part of our Machine Learning Pipeline formative assignment, focusing on building a robust system for climate data management and agricultural yield prediction. The system handles data ingestion, storage, retrieval, and manipulation across two databases, serving as a foundation for machine learning applications.

## 🚀 Features

- **Dual Database Support**: MySQL and MongoDB for data redundancy
- **Data Retrieval**: Fetch climate data records with configurable limits
- **Data Creation**: Add new climate data entries to both databases
- **Machine Learning**: Predict agricultural yields based on climate data
- **Error Handling**: Comprehensive exception handling and logging
- **Type Safety**: Full type hints for better code maintainability

## 🛠️ Technology Stack

- **Backend Frameworks**:
  - FastAPI with SQLAlchemy (MySQL API)
  - FastAPI with Motor (MongoDB API)
- **Databases**:
  - MySQL
  - MongoDB
- **Machine Learning**:
  - TensorFlow
  - NumPy
- **Python**: 3.8+
- **Environment**: Python virtual environment

## 🚀 Running the System

### Prerequisites

1. Python 3.8 or higher
2. MongoDB installed and running
3. MySQL installed and running
4. Virtual environment created and activated

### Step 1: Install Dependencies

```bash
# Install MySQL API dependencies
cd api
pip install -r requirements.txt

# Install MongoDB API dependencies
cd ../mongo_api
pip install -r requirements.txt

# Install prediction dependencies
cd ../prediction
pip install -r requirements.txt
```

### Step 2: Start the APIs

Open two separate terminal windows and run:

1. MySQL API (Terminal 1):

```bash
cd api
uvicorn main:app --reload --port 8000
```

2. MongoDB API (Terminal 2):

```bash
cd mongo_api
uvicorn main:app --reload --port 8001
```

The APIs will be available at:

- MySQL API: http://127.0.0.1:8000
- MongoDB API: http://127.0.0.1:8001

### Step 3: Run the Prediction System

There are two ways to use the prediction system:

1. Fetch data only:

```bash
cd prediction
python fetch_data.py
```

This will fetch the latest records from both APIs and compare them.

2. Make predictions:

```bash
cd prediction
python predict.py
```

This will:

- Fetch the latest climate data
- Run it through the ML model
- Generate a yield prediction
- Save the results locally

### 📝 Expected Outputs

- `fetch_data.py` will create a `data_comparison.json` file showing differences between MySQL and MongoDB data
- `predict.py` will create a `prediction_output.json` file with prediction results

### ⚠️ Troubleshooting

1. If Python is not found, use `py` instead of `python`:

   ```bash
   py fetch_data.py
   py predict.py
   ```

2. If APIs are not reachable:

   - Verify both API servers are running
   - Check the ports (8000 for MySQL, 8001 for MongoDB)
   - Ensure MongoDB service is running
   - Ensure MySQL service is running

3. If prediction fails:
   - Verify TensorFlow is installed
   - Check if the model file exists at the specified path
   - Ensure all required packages are installed

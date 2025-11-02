from pymongo import MongoClient

def get_database():
    # MongoDB connection URL - using default localhost and port
    client = MongoClient('mongodb://localhost:27017/')
    
    # Return the database instance
    return client['agricultural_yields']  # database name
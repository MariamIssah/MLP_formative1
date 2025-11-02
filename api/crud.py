from sqlalchemy.orm import Session
from models import ClimateData

def get_all_data(db: Session):
    """
    Retrieve all climate data records from the database with a limit of 10.
    
    Args:
        db (Session): SQLAlchemy database session
        
    Returns:
        List[ClimateData]: List of ClimateData objects
    """
    return db.query(ClimateData).limit(10).all()

def create_data(db: Session, data):
    """
    Create a new climate data record in the database.
    
    Args:
        db (Session): SQLAlchemy database session
        data (dict): Dictionary containing climate data fields
        
    Returns:
        ClimateData: Newly created ClimateData object
    """
    new_data = ClimateData(**data)
    db.add(new_data)
    db.commit()
    db.refresh(new_data)
    return new_data

def update_data(db: Session, data_id: int, new_data):
    """
    Update an existing climate data record.
    
    Args:
        db (Session): SQLAlchemy database session
        data_id (int): ID of the record to update
        new_data (dict): Dictionary containing fields to update
        
    Returns:
        ClimateData: Updated ClimateData object or None if not found
    """
    data = db.query(ClimateData).filter(ClimateData.id == data_id).first()
    if data:
        for key, value in new_data.items():
            setattr(data, key, value)
        db.commit()
        db.refresh(data)
    return data

def delete_data(db: Session, data_id: int):
    """
    Delete a climate data record from the database.
    
    Args:
        db (Session): SQLAlchemy database session
        data_id (int): ID of the record to delete
        
    Returns:
        ClimateData: Deleted ClimateData object or None if not found
    """
    data = db.query(ClimateData).filter(ClimateData.id == data_id).first()
    if data:
        db.delete(data)
        db.commit()
    return data

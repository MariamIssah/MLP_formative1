from sqlalchemy.orm import Session
from models import ClimateData

def get_all_data(db: Session):
    return db.query(ClimateData).limit(10).all()

def create_data(db: Session, data):
    new_data = ClimateData(**data)
    db.add(new_data)
    db.commit()
    db.refresh(new_data)
    return new_data

def update_data(db: Session, data_id: int, new_data):
    data = db.query(ClimateData).filter(ClimateData.id == data_id).first()
    if data:
        for key, value in new_data.items():
            setattr(data, key, value)
        db.commit()
        db.refresh(data)
    return data

def delete_data(db: Session, data_id: int):
    data = db.query(ClimateData).filter(ClimateData.id == data_id).first()
    if data:
        db.delete(data)
        db.commit()
    return data
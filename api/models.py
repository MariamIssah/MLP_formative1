from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class Country(Base):
    __tablename__ = "countries"

    country_id = Column(Integer, primary_key=True, index=True)
    country_name = Column(String(100), unique=True, nullable=False)

    # A country can have many climate data records
    climate_data = relationship("ClimateData", back_populates="country")

    def __repr__(self):
        return f"<Country(id={self.country_id}, name='{self.country_name}')>"


class Crop(Base):
    __tablename__ = "crops"

    crop_id = Column(Integer, primary_key=True, index=True)
    crop_name = Column(String(100), unique=True, nullable=False)

    # A crop can have many climate data records
    climate_data = relationship("ClimateData", back_populates="crop")

    def __repr__(self):
        return f"<Crop(id={self.crop_id}, name='{self.crop_name}')>"


class ClimateData(Base):
    __tablename__ = "climate_data"

    id = Column(Integer, primary_key=True, index=True)
    country_id = Column(Integer, ForeignKey("countries.country_id"))
    crop_id = Column(Integer, ForeignKey("crops.crop_id"))
    year = Column(Integer, nullable=False)
    avg_temp = Column(Float)
    hg_ha_yield = Column(Float)

    # Relationships
    country = relationship("Country", back_populates="climate_data")
    crop = relationship("Crop", back_populates="climate_data")

    def __repr__(self):
        return (
            f"<ClimateData(id={self.id}, country_id={self.country_id}, "
            f"crop_id={self.crop_id}, year={self.year}, "
            f"avg_temp={self.avg_temp}, yield={self.hg_ha_yield})>"
        )
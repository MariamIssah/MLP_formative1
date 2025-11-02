"""
Pydantic Data Models for Crop Yield Analysis System

This module defines the data models used for storing and validating crop yield data
in MongoDB. It includes custom type handling for MongoDB ObjectId and the main
YieldModel for agricultural data.

"""

from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId
from pydantic_core import core_schema  


# Custom ObjectId type for Pydantic v2 
# This class enables proper validation and serialization of MongoDB ObjectIds
# in Pydantic v2, which doesn't have built-in ObjectId support
class PyObjectId(str):
    """
    Custom Pydantic type for MongoDB ObjectId validation and serialization.
    
    This class ensures that:
    1. ObjectId strings are properly validated
    2. ObjectIds are serialized as strings in JSON
    3. OpenAPI documentation correctly represents ObjectId as string type
    """
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Define how to validate ObjectId - uses Pydantic v2's core schema
        return core_schema.no_info_after_validator_function(
            cls.validate, core_schema.str_schema()
        )

    @classmethod
    def validate(cls, v):
        """
        Validate that the input is a valid MongoDB ObjectId.
        
        Args:
            v: The value to validate
            
        Returns:
            str: The validated ObjectId as string
            
        Raises:
            ValueError: If the input is not a valid ObjectId
        """
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema_, handler):
        """
        Custom JSON schema generation for OpenAPI documentation.
        
        Ensures that ObjectId is represented as a string type in API documentation,
        making it clear to API consumers what format to expect.
        """
        json_schema = handler(core_schema_)
        json_schema.update(type="string")
        return json_schema


# MongoDB Data Model 
class YieldModel(BaseModel):
    """
    Main data model representing crop yield and agricultural metrics.
    
    This model maps to MongoDB documents and includes validation for all fields.
    It's used for both API request/response validation and database operations.
    
    Fields:
    - id: MongoDB ObjectId (automatically generated)
    - country: Country name where data was recorded
    - crop: Type of crop (e.g., Maize, Wheat, Rice)
    - year: Year of data recording
    - yield_hg_per_ha: Crop yield in hectograms per hectare
    - average_rainfall_mm_per_year: Annual rainfall in millimeters
    - pesticides_tonnes: Pesticide usage in tonnes
    - avg_temp: Average temperature in Celsius
    """
    
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    country: str
    crop: str
    year: int
    yield_hg_per_ha: float = Field(..., description="Crop yield in hectograms per hectare")
    average_rainfall_mm_per_year: float = Field(..., description="Annual rainfall in millimeters")
    pesticides_tonnes: float = Field(..., description="Pesticide usage in tonnes")
    avg_temp: float = Field(..., description="Average temperature in Celsius")

    # Pydantic v2 configuration
    model_config = {
        "populate_by_name": True,  # Allows using both field name and alias
        "json_encoders": {ObjectId: str},  # Convert ObjectId to string in JSON
        "json_schema_extra": {  # Example schema for API documentation
            "example": {
                "country": "Kenya",
                "crop": "Maize",
                "year": 2023,
                "yield_hg_per_ha": 4500.75,
                "average_rainfall_mm_per_year": 800.0,
                "pesticides_tonnes": 20.5,
                "avg_temp": 24.7
            }
        }
    }

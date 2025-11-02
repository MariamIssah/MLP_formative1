from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId
from pydantic_core import core_schema  

class PyObjectId(str):

    
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
    
        json_schema = handler(core_schema_)
        json_schema.update(type="string")
        return json_schema


# MongoDB Data Model 
class YieldModel(BaseModel):
    
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

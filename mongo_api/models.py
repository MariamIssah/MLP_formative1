from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId
from pydantic_core import core_schema  


# ---- Custom ObjectId type for Pydantic v2 ----
class PyObjectId(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Define how to validate ObjectId
        return core_schema.no_info_after_validator_function(
            cls.validate, core_schema.str_schema()
        )

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema_, handler):
        # Ensure OpenAPI docs treat ObjectId as string
        json_schema = handler(core_schema_)
        json_schema.update(type="string")
        return json_schema


# ---- MongoDB Data Model ----
class YieldModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    country: str
    crop: str
    year: int
    yield_hg_per_ha: float
    average_rainfall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float

    model_config = {
        "populate_by_name": True,
        "json_encoders": {ObjectId: str},
        "json_schema_extra": {  # <— renamed for Pydantic v2
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

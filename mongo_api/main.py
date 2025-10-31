from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from models import YieldModel

app = FastAPI(title="Farm Yields API", version="1.0")

# -----------------------------
# MongoDB connection settings
# -----------------------------
client = AsyncIOMotorClient("mongodb://localhost:27017/")
db = client["farm_yields_database"]
collection = db["yields"]

# -----------------------------
# Create (POST)
# -----------------------------
@app.post("/yields/", response_model=YieldModel)
async def create_yield(yield_data: YieldModel):
    yield_dict = yield_data.model_dump(by_alias=True, exclude={"id"})
    result = await collection.insert_one(yield_dict)
    created = await collection.find_one({"_id": result.inserted_id})
    created["_id"] = str(created["_id"])
    return YieldModel(**created)


# -----------------------------
# Read all (GET)
# -----------------------------
@app.get("/yields/", response_model=list[YieldModel])
async def get_all_yields():
    docs = await collection.find().to_list(100)
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return [YieldModel(**doc) for doc in docs]


# -----------------------------
# Read one (GET by ID)
# -----------------------------
@app.get("/yields/{yield_id}", response_model=YieldModel)
async def get_yield(yield_id: str):
    try:
        record = await collection.find_one({"_id": ObjectId(yield_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    if record is None:
        raise HTTPException(status_code=404, detail="Yield record not found")

    record["_id"] = str(record["_id"])
    return YieldModel(**record)


# -----------------------------
# Update (PUT)
# -----------------------------
@app.put("/yields/{yield_id}", response_model=YieldModel)
async def update_yield(yield_id: str, yield_data: YieldModel):
    update_dict = {
        k: v for k, v in yield_data.model_dump(by_alias=True).items()
        if v is not None and k != "_id"
    }

    try:
        result = await collection.update_one(
            {"_id": ObjectId(yield_id)}, {"$set": update_dict}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Yield record not found")

    updated_record = await collection.find_one({"_id": ObjectId(yield_id)})
    updated_record["_id"] = str(updated_record["_id"])
    return YieldModel(**updated_record)


# -----------------------------
# Delete (DELETE)
# -----------------------------
@app.delete("/yields/{yield_id}")
async def delete_yield(yield_id: str):
    try:
        result = await collection.delete_one({"_id": ObjectId(yield_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Yield record not found")

    return {"message": f"Yield record {yield_id} deleted successfully"}

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from backend.db.database import get_db
from backend.models.models import Brand
import uuid

router = APIRouter(prefix="/api/brands", tags=["brands"])


class BrandCreate(BaseModel):
    name: str
    domain: str
    products: list = []
    customers: list = []
    key_features: list = []
    business_type: str = "SaaS / Software"
    country: str = "United States"


class BrandResponse(BaseModel):
    id: str
    name: str
    domain: str
    products: list
    customers: list
    key_features: list
    business_type: str
    country: str

    class Config:
        from_attributes = True


@router.post("/", response_model=BrandResponse)
def create_brand(data: BrandCreate, db: Session = Depends(get_db)):
    brand = Brand(
        id=str(uuid.uuid4()),
        name=data.name,
        domain=data.domain,
        products=data.products,
        customers=data.customers,
        key_features=data.key_features,
        business_type=data.business_type,
        country=data.country,
    )
    db.add(brand)
    db.commit()
    db.refresh(brand)
    return brand


@router.get("/", response_model=list[BrandResponse])
def list_brands(db: Session = Depends(get_db)):
    return db.query(Brand).all()


@router.get("/{brand_id}", response_model=BrandResponse)
def get_brand(brand_id: str, db: Session = Depends(get_db)):
    brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    return brand


@router.put("/{brand_id}", response_model=BrandResponse)
def update_brand(brand_id: str, data: BrandCreate, db: Session = Depends(get_db)):
    brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    for key, value in data.dict().items():
        setattr(brand, key, value)
    db.commit()
    db.refresh(brand)
    return brand


@router.delete("/{brand_id}")
def delete_brand(brand_id: str, db: Session = Depends(get_db)):
    brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    db.delete(brand)
    db.commit()
    return {"ok": True}

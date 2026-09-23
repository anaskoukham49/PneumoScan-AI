import os
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pneumoscan.db")

if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

USER_MIGRATION_COLUMNS = {
    "blood_type": "VARCHAR",
    "family_has_illness": "BOOLEAN DEFAULT 0",
    "family_illness_history": "TEXT",
    "smokes": "BOOLEAN DEFAULT 0",
    "current_illness": "TEXT",
    "avg_heartbeat": "INTEGER",
}


def migrate_schema() -> None:
    """Add missing columns/tables for copied projects with older databases."""
    from api import models  # noqa: F401

    models.Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("users")}
    with engine.begin() as conn:
        for column, col_type in USER_MIGRATION_COLUMNS.items():
            if column not in existing:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {column} {col_type}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

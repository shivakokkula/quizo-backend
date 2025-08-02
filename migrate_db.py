from app import Base, engine

def migrate_database():
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Creating all tables with latest schema...")
    Base.metadata.create_all(bind=engine)
    print("Database migration completed successfully!")

if __name__ == "__main__":
    migrate_database()

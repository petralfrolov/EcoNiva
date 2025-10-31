# database.py

import bcrypt
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from contextlib import contextmanager  # <-- 1. ДОБАВЬТЕ ЭТОТ ИМПОРТ

# --- Настройка БД ---
DATABASE_URL = "sqlite:///users.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Список подразделений ---
DEPARTMENTS = ['ЖК Высокое', 'ЖK Бобров', 'ЖК Петропавловка']


# --- Модель пользователя ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    fio = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    department = Column(String, nullable=False)


# --- Функции для работы с БД ---

def init_db():
    """Создает все таблицы в базе данных."""
    Base.metadata.create_all(bind=engine)


@contextmanager  # <-- 2. ДОБАВЬТЕ ЭТОТ ДЕКОРАТОР
def get_db():
    """Генератор сессии БД."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Функции для работы с пользователями ---

def hash_password(password: str) -> str:
    """Хеширует пароль."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def check_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет соответствие пароля хешу."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def create_user(fio: str, username: str, password: str, department: str) -> User | None:
    """Создает нового пользователя в БД."""
    hashed = hash_password(password)
    new_user = User(
        fio=fio,
        username=username,
        hashed_password=hashed,
        department=department
    )

    with get_db() as db:  # <--- Теперь эта строка будет работать
        try:
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            return new_user
        except IntegrityError:
            db.rollback()
            return None  # Пользователь уже существует
        except Exception:
            db.rollback()
            raise


def get_user(username: str) -> User | None:
    """Получает пользователя по его логину."""
    with get_db() as db:  # <--- И эта (проблемная) строка теперь будет работать
        return db.query(User).filter(User.username == username).first()
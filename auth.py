import bcrypt
from database import users_collection


def create_user(username, password):
    # minimum 6 chars
    if len(password) < 6:
        return "Password must be at least 6 characters"

    if users_collection.find_one({"username": username}):
        return "User already exists"

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    users_collection.insert_one({
        "username": username,
        "password": hashed_pw
    })

    return "User created"


def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})

    if not user:
        return False

    stored_pw = user["password"]

    # krahaso password me hash
    return bcrypt.checkpw(password.encode("utf-8"), stored_pw)
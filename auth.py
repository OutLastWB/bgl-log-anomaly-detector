from database import users_collection

def create_user(username, password):
    if users_collection.find_one({"username": username}):
        return False
    
    users_collection.insert_one({
        "username": username,
        "password": password
    })
    
    return True


def authenticate_user(username, password):
    user = users_collection.find_one({
        "username": username,
        "password": password
    })
    
    return user is not None
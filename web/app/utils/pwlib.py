import bcrypt


def encrypt_password(password: bytes):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password, salt)

    return hashed_password, salt


def check_password(password: bytes, hashed_password: bytes, salt: bytes):
    new_hashed_password = bcrypt.hashpw(password, salt)

    return new_hashed_password == hashed_password

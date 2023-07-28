import os

import mysql.connector
from app.schema import Pet, Script, User
from dotenv import load_dotenv

load_dotenv()


class SQLConnector:
    def __init__(self):
        self.cnx = mysql.connector.connect(
            host=os.environ["DB_HOST"],
            user=os.environ["DB_USER"],
            port=os.environ["DB_PORT"],
            password=os.environ["DB_PASS"],
            database=os.environ["DB_DATABASE"],
        )
        self.cursor = self.cnx.cursor()

    def __enter__(self, *args):
        return self.cursor

    def __exit__(self, *args):
        self.cnx.commit()
        self.cursor.close()
        self.cnx.close()


def get_user(id):
    with SQLConnector() as cursor:
        query = f"SELECT * FROM user WHERE id='{id}'"
        cursor.execute(query)
        try:
            result = User(*next(cursor))
        except:
            return None

    return result


def check_id(id):
    with SQLConnector() as cursor:
        query = f"SELECT * FROM user WHERE id='{id}'"
        cursor.execute(query)
        try:
            next(cursor)
        except:
            return False
    return True


def insert_user(id, pw, salt):
    if not check_id(id):
        with SQLConnector() as cursor:
            query = f"INSERT INTO user VALUES('{id}', '{pw}', '{salt}')"
            cursor.execute(query)

        result = get_user(id)
        if result is not None:
            return "성공"
        else:
            return "실패"
    else:
        return "ID중복"


def check_pet(id):
    with SQLConnector() as cursor:
        query = f"SELECT * FROM pet WHERE id='{id}'"
        cursor.execute(query)
        try:
            next(cursor)
        except:
            return False
    return True


def insert_pet(id, name, birth, catordog, gender):
    if not check_pet(id):
        with SQLConnector() as cursor:
            query = f"INSERT INTO pet VALUES('{id}', '{name}', '{birth}', '{catordog}', '{gender}')"
            cursor.execute(query)

        if check_pet(id):
            return "성공"
        else:
            return "실패"
    else:
        return "이미 등록된 펫"


def get_pet(id):
    with SQLConnector() as cursor:
        query = f"SELECT * FROM pet WHERE id='{id}'"
        cursor.execute(query)
        try:
            result = Pet(*next(cursor))
        except:
            return None

    return result


def get_script(cat_dog, part, desease):
    with SQLConnector() as cursor:
        query = f"SELECT * FROM script WHERE (class='{cat_dog}' AND part='{part}' AND desease='{desease}')"
        cursor.execute(query)
        try:
            result = Script(*next(cursor))
        except:
            return None

    return result

class User:
    def __init__(self, id: str, pw: str, salt: str):
        self.id: str = id
        self.pw: str = pw
        self.salt: str = salt

EN_PUNCTUATION = r".,;:/\\|<>'\"[]{}!@#$%^&*()?-_=+`~"
CN_PUNCTUATION = r"。？！，…「」（）《》"
PUNCTUATION_SPLIT = EN_PUNCTUATION + CN_PUNCTUATION


def token_split(phrase: str):
    tokens = []
    buf = ""
    for char in phrase:
        if char == " ":
            tokens.append(buf)
            buf = char
        elif char in PUNCTUATION_SPLIT:
            tokens.append(buf)
            tokens.append(char)
            buf = ""
        else:
            buf += char
    tokens.append(buf)
    return tokens

# doubly linked list implementation


class Node:
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

# for easier handling of bytes


class ByteSeq:
    def __init__(self, data):
        self.data = data if isinstance(data, bytes) else bytes([data])

    def __add__(self, other):
        return ByteSeq(self.data + other.data)

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __lt__(self, other):
        return self.data < other.data

    def __str__(self):
        return self.data.decode("utf-8", errors="replace")

    def __repr__(self):
        return self.__str__()

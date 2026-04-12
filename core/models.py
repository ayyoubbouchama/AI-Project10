class Box:
    def __init__(self, id, l, w, h, weight=1):
        self.id = id
        self.l = l
        self.w = w
        self.h = h
        self.weight = weight


class Container:
    def __init__(self, L, W, H):
        self.L = L
        self.W = W
        self.H = H
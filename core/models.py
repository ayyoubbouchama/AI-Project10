class Box:
    def __init__(self, id, l, w, h, weight=1):
        self.id = id
        self.l = l
        self.w = w
        self.h = h
        self.weight = weight

    def volume(self):
        return self.l * self.w * self.h

    def __repr__(self):
        return f"Box(id={self.id}, {self.l}x{self.w}x{self.h}, weight={self.weight})"


class Container:
    def __init__(self, L, W, H):
        self.L = L
        self.W = W
        self.H = H

    def volume(self):
        return self.L * self.W * self.H

    def __repr__(self):
        return f"Container({self.L}x{self.W}x{self.H})"
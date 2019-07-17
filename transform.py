from PIL import Image
import collections

class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)








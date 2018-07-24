from weakref import WeakValueDictionary

class Counter:
    _instances = WeakValueDictionary()
    @property
    def Count(self):
        return len(self._instances)

    def __init__(self, name):
        self.name = name
        self._instances[id(self)] = self
        print(name, 'created')

    def __del__(self):
        print(self.name, 'deleted')
        if self.Count == 0:
            print('Last Counter object deleted')
        else:
            print(self.Count, 'Counter objects remaining')

x = Counter("First")


class Counter:
    Count = 0   # This represents the count of objects of this class
    def __init__(self, name):
        self.name = name
        print(name, 'created')
        Counter.Count += 1
    def __del__(self):
        print(self.name, 'deleted')
        Counter.Count -= 1
        if Counter.Count == 0:
            print('Last Counter object deleted')
        else:
            print(Counter.Count, 'Counter objects remaining')

x = Counter("First")
del x
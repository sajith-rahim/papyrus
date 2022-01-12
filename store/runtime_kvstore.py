class RuntimeKeyValueStore:

    def get_item(self, key):
        if hasattr(self, key):
            value = getattr(self, key)
            return value
        else:
            raise AttributeError("key not found.")

    def set_item(self, key, val, overwrite=True):
        if not overwrite and hasattr(self, key):
            raise AttributeError("key already exists and overwrite not allowed.")
        else:
            setattr(self, key, val)

    def set_all(self, overwrite=True, **kv):
        for key, val in kv.items():
            self.set_item(key, val, overwrite)

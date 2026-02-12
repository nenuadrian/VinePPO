import os
import shutil
import sqlite3

import diskcache
import platformdirs

from guidance.llms.caches import Cache


class DiskCache(Cache):
    """DiskCache is a cache that uses diskcache lib."""
    def __init__(self, llm_name: str):
        cache_dir = os.path.join(
            platformdirs.user_cache_dir("guidance"), f"_{llm_name}.diskcache"
        )
        try:
            self._diskcache = diskcache.Cache(cache_dir)
        except sqlite3.DatabaseError as exc:
            # Recover from corrupted cache DBs produced by interrupted runs.
            if "file is not a database" not in str(exc):
                raise
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            elif os.path.exists(cache_dir):
                os.remove(cache_dir)
            self._diskcache = diskcache.Cache(cache_dir)

    def __getitem__(self, key: str) -> str:
        return self._diskcache[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._diskcache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._diskcache
    
    def clear(self):
        self._diskcache.clear()

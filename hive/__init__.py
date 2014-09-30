import asyncio
import aiohs2
import pandas as pd

from functools import wraps
coroutine = asyncio.coroutine

class AioHive:
    def __init__(self, host, port=10000):
        """
        coroutine based hive client

        Parameters
        ==========
        host : str
            host of the hiveserver2 to connect to
        port : int, default 10000
            port of the hiveserver2
        """
        self.cli = aiohs2.Client(host=host, port=port)

    def execute(self, request):
        """ execute request without looking at returns """
        yield from self.cli.execute(request)

    def fetch(self, hql, chunk_size=10000):
        """ execute request and fetch answer as DataFrame """
        with (yield from self.cli.cursor()) as cur:
            yield from cur.execute(hql)
            schema = yield from cur.getSchema()
            columns = pd.Index([nfo['columnName'] for nfo in schema])

            return pd.DataFrame((yield from cur.fetch(maxRows=chunk_size)) or None, columns=columns)


    def iter(self, hql, chunk_size=10000):
        """ execute request and iterate over chunks of resulting DataFrame """
        cur = yield from self.cli.cursor()

        try:
            yield from cur.execute(hql)
            schema = yield from cur.getSchema()
            columns = pd.Index([nfo['columnName'] for nfo in schema])

            chunks = cur.iter(maxRows=chunk_size)

            class local:
                offset=0
                empty=None

            @coroutine
            def to_frame(chunk_co):
                data = pd.DataFrame((yield from chunk_co) or local.empty, columns=columns)
                data.index += local.offset

                local.offset += len(data)
                if local.empty is None:
                    local.empty = data[:0].copy()
                return data

            def closing():
                try:
                    for chunk in chunks:
                        # here we yield the coroutine that will fetch the data and put in in a frame
                        yield to_frame(chunk)
                finally:
                    # while ensuring that the cursor is closed after the request is done ....
                    cur.close()

            return closing()

        except:
            cur.close()
            raise

class SyncedHive:
    def __init__(self, *args, hive=None, **kws):
        """
        synced wrapper around the asyncio hive class

        Parameters
        ==========
        host : str
            host of the hiveserver2 to connect to
        port : int, default 10000
            port of the hiveserver2
        hive : AioHive, optional
            existing async hive client
        """
        self.hive = hive or AioHive(*args, **kws)
        self.loop = asyncio.get_event_loop()

    def run(self, coro):
        return self.loop.run_until_complete(coro)

    def synced(name):
        func = getattr(AioHive, name)
        @wraps(func)
        def synced(self, *args, **kws):
            return self.run(func(self.hive, *args, **kws))
        return synced

    execute = synced('execute')
    fetch = synced('fetch')

    def iter(self, *args, **kws):
        for chunk in self.run(self.hive.iter(*args, **kws)):
            yield self.run(chunk)


Hive = SyncedHive

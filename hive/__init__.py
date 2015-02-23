import asyncio
import aiohs2
import pandas as pd

import logging

from functools import wraps
coroutine = asyncio.coroutine

logger = logging.getLogger(__name__)

hive_type_map = {
    'BOOLEAN': pd.np.dtype(bool),
    'BINARY': pd.np.dtype(bytes),

    'TINYINT': pd.np.dtype(int),
    'SMALLINT': pd.np.dtype(int),
    'INT': pd.np.dtype(int),
    'BIGINT': pd.np.dtype(int),
    'FLOAT': pd.np.dtype(float),
    'DOUBLE': pd.np.dtype(float),
    'DECIMAL': pd.np.dtype(float),

    'TIMESTAMP': pd.np.dtype('datetime64'),
    'DATE':      pd.np.dtype('datetime64'),

    'STRING': pd.np.dtype(str),
    'VARCHAR': pd.np.dtype(str),
    'CHAR': pd.np.dtype(str),

    'ARRAY': pd.np.dtype(list),
    'MAP': pd.np.dtype(dict),
    'STRUCT': pd.np.dtype(object),
    'UNIONTYPE': pd.np.dtype(object),
}


class AioHive:
    def __init__(self, host=None, config=None, port=10000):
        """
        coroutine based hive client

        Parameters
        ==========
        host : str
            host of the hiveserver2 to connect to
        port : int, default 10000
            port of the hiveserver2
        """
        if (host is None and config is None) or (config and host):
            raise TypeError('Either host or config argument has to be supplied')
        if config:
            import xml.etree.ElementTree as ET
            cfg = ET.parse(config)
            for res in cfg.iter('property'):
                if res.findtext('name') == 'hive.metastore.uris':
                    uri = res.findtext('value')
                    host = uri.split('://')[-1].split(':')[0]
                    break
            else:
                raise ValueError(
                    "could not find 'hive.metastore.uris' in config")
        self.cli = aiohs2.Client(host=host, port=port)

    @coroutine
    def execute(self, request):
        """ execute request without looking at returns """
        cur = yield from self.cli.cursor()
        try:
            yield from cur.execute(request)
        finally:
            yield from cur.close()

    @staticmethod
    def get_dtype(typ):
        try:
            return hive_type_map[typ.rsplit('<', 1)[0].rsplit('_', 1)[0]]
        except KeyError:
            logger.warning('Unknown type %r for hive request', typ)
            return pd.np.dtype(object)

    @coroutine
    def fetch(self, hql, chunk_size=10000):
        """ execute request and fetch answer as DataFrame """
        cur = yield from self.cli.cursor()
        try:
            yield from cur.execute(hql)
            schema = yield from cur.getSchema()
            columns = pd.Index([nfo['columnName'] for nfo in schema])
            dtypes = [self.get_dtype(nfo['type']) for nfo in schema]

            data = (yield from cur.fetch(maxRows=chunk_size)) or None
            df = pd.DataFrame(data, columns=columns, dtype=object)
            for col, typ in zip(columns, dtypes):
                if typ == pd.np.dtype('datetime64') and df[col].isnull().all():
                    df[col] = pd.NaT
                else:
                    try:
                        df[col] = df[col].astype(typ)
                    except TypeError as e:
                        logger.warning('Cannot convert %r to %r (%s)',
                                       col, typ, e)
            return df
        finally:
            yield from cur.close()

    def iter(self, hql, chunk_size=10000):
        """ execute request and iterate over chunks of resulting DataFrame """
        cur = yield from self.cli.cursor()

        try:
            yield from cur.execute(hql)
            schema = yield from cur.getSchema()
            columns = pd.Index([nfo['columnName'] for nfo in schema])
            dtypes = [self.get_dtype(nfo['type']) for nfo in schema]

            chunks = cur.iter(maxRows=chunk_size)

            class local:
                offset = 0
                empty = None
                warns = set()

            @coroutine
            def to_frame(chunk_co):
                df = pd.DataFrame((yield from chunk_co) or local.empty,
                                  columns=columns, dtype=object)
                df.index += local.offset

                local.offset += len(df)
                if local.empty is None:
                    local.empty = df[:0].copy()
                for col, typ in zip(columns, dtypes):
                    try:
                        df[col] = df[col].astype(typ)
                    except TypeError as e:
                        if col not in local.warns:
                            logger.warning('Cannot convert %r to %r (%s)',
                                        col, typ, e, exc_info=True)
                            local.warns.add(col)
                return df

            def closing():
                try:
                    for chunk in chunks:
                        # here we yield the coroutine that will fetch the data
                        # and put in in a frame
                        yield to_frame(chunk)
                finally:
                    # while ensuring that the cursor is closed ...
                    cur.close()

            return closing()

        finally:
            cur.close()


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
            data = self.run(chunk)
            if not data.empty:
                yield data

Hive = SyncedHive

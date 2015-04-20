import asyncio
import aiohs2
import pandas as pd
import subprocess

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


class Framer:
    def __init__(self, columns, dtypes, fill_values=None):
        self.columns = columns
        self.dtypes = dtypes
        self.offset = 0
        self.fill_values = fill_values or {}
        self.warns = set()

    @staticmethod
    def get_dtype(typ):
        try:
            return hive_type_map[typ.rsplit('<', 1)[0].rsplit('_', 1)[0]]
        except KeyError:
            logger.warning('Unknown type %r for hive request', typ)
            return pd.np.dtype(object)

    @classmethod
    @coroutine
    def by_cursor(cls, cur, hql, **kws):
        yield from cur.execute(hql)
        schema = (yield from cur.getSchema())
        if schema is None:
            columns = dtypes = None
        else:
            columns = pd.Index([nfo['columnName'] for nfo in schema])
            dtypes = [cls.get_dtype(nfo['type']) for nfo in schema]
        return cls(columns, dtypes, **kws)

    @coroutine
    def __call__(self, coro):
        raw = yield from coro
        return self.mk_df(raw)

    def mk_df(self, raw, na_val=None):
        if self.columns is None:
            if raw is None:
                return None
            else:
                if raw is not None and '__schema__' not in self.warns:
                    logger.warning('no schema, but got data from hive')
                    self.warns.add('__schema__')

                return pd.DataFrame(raw, dtype=object)
        df = pd.DataFrame(raw or None,  # self.empty,
                          columns=self.columns, dtype=object)
        if na_val is not None:
            df[df == na_val] = None
        df.index += self.offset
        self.offset += len(df)

        for col, val in self.fill_values.items():
            df[col] = df[col].fillna(val)

        # if self.empty is None:
        #     local.empty = df[:0].copy()
        for col, typ in zip(self.columns, self.dtypes):
            try:
                df[col] = df[col].astype(typ)
            except TypeError as e:
                if col not in self.warns:
                    logger.warning('Cannot convert %r to %r (%s)', col, typ, e)
                    self.warns.add(col)
        return df


class RawHDFSChunker:
    def __init__(self, hive, table, partitions):
        self.hive = hive
        self.table = table
        self.partitions = partitions[:]

        self.partition = None
        self.framer = None
        self.proc = None
        self.tail = b''

    @coroutine
    def next_part(self):
        yield from self.close()

        self.partition = self.partitions.pop(0)
        self.framer, self.proc = yield from self.hive._raw_hdfs(
            self.table, self.partition)
        self.tail = b''

    @coroutine
    def chunker(self):
        while True:
            if not self.partition:
                yield from self.next_part()

            chunk = yield from self.proc.stdout.read(24000000)
            if not chunk:
                self.partition = None
                if self.tail:
                    chunk = self.tail
                    self.tail = b''
                    break
                else:
                    continue
            split = (self.tail + chunk).rsplit(b'\n', 1)
            if len(split) == 1:
                self.tail = chunk
            else:
                chunk, self.tail = split
                break

        if chunk:
            raw = [l.split('\t')[1:] for l in chunk.decode().split('\n')]
            return self.framer.mk_df(raw, na_val='')
        else:
            return None

    def iter(self):
        try:
            while True:
                fut = asyncio.async(self.chunker())
                yield fut
                if not fut.result:
                    break
        finally:
            yield self.close()

    @coroutine
    def close(self):
        if self.proc:
            self.proc.send_signal(subprocess.signal.SIGINT)
            yield from self.proc.wait()


class AioHive:
    def __init__(self, host=None, config=None, port=10000):
        """
        coroutine based hive client

        Parameters
        ==========
        host : str
            host of the hiveserver2 to connect to
        config : str
            hive-site.xml to extract hive.metastore.uris
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
        self.config = config

    @coroutine
    def execute(self, *rqs):
        """ execute request without looking at returns """
        cur = yield from self.cli.cursor()
        try:
            for rq in rqs:
                yield from cur.execute(rq)
        finally:
            yield from cur.close()

    @coroutine
    def fetch(self, hql, chunk_size=10000, fill_values=None):
        """ execute request and fetch answer as DataFrame """
        cur = yield from self.cli.cursor()
        try:
            framer = yield from Framer.by_cursor(cur, hql,
                                                 fill_values=fill_values)
            return (yield from framer(cur.fetch(maxRows=chunk_size)))
        finally:
            yield from cur.close()

    def iter(self, hql, chunk_size=10000, fill_values=None):
        """ execute request and iterate over chunks of resulting DataFrame """
        cur = yield from self.cli.cursor()
        framer = yield from Framer.by_cursor(cur, hql,
                                             fill_values=fill_values)
        chunks = cur.iter(maxRows=chunk_size)

        def iter_chunks():
            try:
                for chunk in chunks:
                    # here we yield the coroutine that will fetch the data
                    # and put in in a frame
                    yield framer(chunk)
            finally:
                yield framer(cur.close())

        return iter_chunks()

    @coroutine
    def close(self):
        yield from self.cli.close()

    @coroutine
    def _raw_hdfs(self, table, partition):
        info = (yield from self.fetch(
            'describe formatted {table} partition ({partition})'
            .format(table=table,
                    partition=partition.replace('=', '="')+'"'))).applymap(
                        str.strip)

        i0, i1, *_ = pd.np.flatnonzero(info.col_name == '')
        schema = info[i0+1:i1]
        location = info.query('col_name == "Location:"').data_type.iloc[0]

        columns = schema.col_name
        dtypes = (schema.data_type
                  .str.split('(').str[0].str.upper()
                  .apply(hive_type_map.__getitem__))
        framer = Framer(columns, dtypes)

        proc = yield from asyncio.create_subprocess_exec(
            'hadoop', 'fs', '-text',
            '/'+location.split('://', 1)[1].split('/', 1)[1]+'/*',
            stdout=subprocess.PIPE)

        return framer, proc

    @coroutine
    def raw(self, table, **partitions):
        parts = yield from self.fetch('show partitions {}'.format(table))
        parts = parts.partition

        info = parts.str.split('=')
        names = info.str[0]
        vals = info.str[1]

        sel = pd.Series(False, index=parts.index)
        for name, val in partitions.items():
            if name not in names.values:
                raise KeyError('no partition info {} in {}', name, table)
            if isinstance(val, str):
                val = [val]
            for v in val:
                sel |= (names == name) & (vals.str.contains(v))

        rhc = RawHDFSChunker(self, table, list(parts[sel]))
        return rhc.iter()


class SyncedHive:
    def __init__(self, *args, hive=None, **kws):
        """
        synced wrapper around the asyncio hive class

        Parameters
        ==========
        host : str
            host of the hiveserver2 to connect to
        config : str
            hive-site.xml to extract hive.metastore.uris
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
    close = synced('close')

    def iter(self, *args, **kws):
        it = self.run(self.hive.iter(*args, **kws))
        try:
            for chunk in it:
                data = self.run(chunk)
                if data is not None and not data.empty:
                    yield data
        except BaseException as e:
            # ensure close is run
            self.run(it.throw(e))
            raise e

    def raw(self, *args, **kws):
        it = self.run(self.hive.raw(*args, **kws))
        try:
            for chunk in it:
                data = self.run(chunk)
                if data is not None and not data.empty:
                    yield data
        except BaseException as e:
            # ensure close is run
            self.run(it.throw(e))
            raise e


Hive = SyncedHive

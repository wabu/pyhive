The pyhive package defines easy to use interface for hive yielding pandas DataFrames.
It directly connects to a hiveserver2 using Thrift/aio-hs2.

Installation
============
You can use pip to install the package:
```
pip install git+https://github.com/wabu/pyhive.git
```

Or you download/clone the package with git and use the setup script:
```
./setup.py develop
```

Usage
=====
```
from hive import Hive
hive = Hive('hiveserver')
hive.execute('use foobar')  # command without getting results
hive.fetch('show tables')   # get results from command as dataframe
# iterate over long result, getting chunks of data
for chunk in hive.iter('select * from baz limit 1000000'):
    print(chunk.tail())
```

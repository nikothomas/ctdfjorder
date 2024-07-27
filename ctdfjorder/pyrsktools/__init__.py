#! /usr/bin/env python3

"""
pyRSKTools is a simple Python toolbox to open RSK SQLite files generated
by RBR instruments.
"""

from collections import namedtuple, OrderedDict
from datetime import datetime, timezone
import itertools
import re
import sqlite3

import ctdfjorder.pyrsktools.channel_types as channel_types

INSTRUMENT_TIME_MIN = datetime(2000, 1, 1, tzinfo=timezone.utc)
INSTRUMENT_TIME_MAX = datetime(2100, 1, 1, tzinfo=timezone.utc)

__copyright__ = "Copyright (c) 2017–2019 RBR Ltd"


class Channel(
    namedtuple("Channel", ["id", "key", "label", "name", "units", "derived"])
):
    __doc__ = """
Each physical sensor on an instrument is represented by one or more logical
channels.
"""

    def label(self):
        return "%s (%s)" % (self.name, self.units)


Deployment = namedtuple(
    "Deployment",
    [
        "id",
        "comment",
        "logger_status",
        "logger_time_drift",
        "download_time",
        "name",
        "sample_size",
    ],
)
Deployment.__doc__ = """
The RSK represents a deployment of the instrument.
"""

Instrument = namedtuple(
    "Instrument", ["serial", "model", "firmware_version", "firmware_type"]
)
Instrument.__doc__ = """
Instruments collect data.
"""

Geo = namedtuple(
    "Geo", ["timestamp", "latitude", "longitude", "accuracy", "accuracy_type"]
)
Geo.__doc__ = """
Geographic position data. May or may not be present in a dataset, depending on
the mode of collection.
"""


def auto_repr(cls):
    def __repr__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join(
                "%s=%r" % var
                for var in vars(self).items()
                if not var[0].startswith("_")
            ),
        )

    cls.__repr__ = __repr__
    return cls


@auto_repr
class RSK(object):
    """
    An RSK dataset produced by Ruskin.
    """

    def __init__(self, rsk):
        # The deployment name gives the name of the dataset at time of download,
        # not the filename we're using to access it right now. Keep the current
        # name around in case someone needs it.
        self.name = rsk
        self._db = sqlite3.connect("file:%s?mode=ro" % rsk, uri=True)

        # Retrieve channels from the dataset.
        self.channels = OrderedDict()
        used_labels = {}
        for row in self._db.execute(
            """select channelID,
                                              shortName,
                                              longName,
                                              units,
                                              isDerived
                                         from channels
                                     order by channelId asc"""
        ):
            channel_id, key, name, units, derived = row
            derived = bool(derived)

            # RSKs don't store channel labels (and L2 channels don't have
            # labels naturally), so we'll generate them the same way Ruskin
            # otherwise would.
            label_prefix = channel_types.label_prefixes.get(
                key, name.lower().replace(" ", "")
            )
            label_counter = used_labels.get(label_prefix, 0)
            used_labels[label_prefix] = label_counter + 1
            label = "%s_%02d" % (label_prefix, label_counter)

            channel = Channel(channel_id, key, label, name, units, derived)
            self.channels[label] = channel

        # Mobile versions of Ruskin don't calculate derived channels, so RSKs
        # they generate won't have a full complement of data columns. We'll work
        # out which are present in the dataset we're dealing with so we can
        # offer the user convenient field names for samples.
        self.sample_channels = [
            column
            for column in _column_names(self._db, "data")
            if column.startswith("channel")
        ]
        self.sample_fields = ["timestamp"] + [
            channel
            for channel in self.channels
            if "channel%02d" % self.channels[channel].id in self.sample_channels
        ]
        self.Sample = namedtuple("Sample", self.sample_fields)

        raw_deployment = self._db.execute(
            """select deploymentID,
                                                    comment,
                                                    loggerStatus,
                                                    loggerTimeDrift,
                                                    timeOfDownload,
                                                    name,
                                                    sampleSize
                                               from deployments
                                              limit 1"""
        ).fetchone()
        logger_time_drift = raw_deployment[3] if raw_deployment[3] is not None else 0
        download_time = datetime.fromtimestamp(
            raw_deployment[4] / 1000, tz=timezone.utc
        )
        self.deployment = Deployment(
            *raw_deployment[:3]
            + (logger_time_drift, download_time)
            + raw_deployment[5:]
        )

        # Some older RSK versions kept the `firmwareVersion` and `firmwareType`
        # fields on the `deployments` table; newer versions keep them in the
        # `instruments` table. We'll sneakily use a join to pull the field from
        # wherever it exists. We don't need to join _on_ anything because there
        # should only be one row in each table (and the `serialID` field may
        # not exist on `deployments`).
        instrument = self._db.execute(
            """select instruments.serialID,
                                                model,
                                                firmwareVersion
                                           from instruments, deployments
                                          limit 1"""
        ).fetchone()

        # Much older RSKs don't have `firmwareType` at all, so we'll go looking
        # for that separately.
        if "firmwareType" in itertools.chain(
            _column_names(self._db, "instruments"),
            _column_names(self._db, "deployments"),
        ):
            firmware_type = self._db.execute(
                """select firmwareType
                                                  from instruments, deployments
                                                 limit 1"""
            )
            instrument = itertools.chain(instrument, firmware_type)
        else:
            instrument = itertools.chain(instrument, (None,))

        self.instrument = Instrument(*instrument)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def geodata(self, start_time=INSTRUMENT_TIME_MIN, end_time=INSTRUMENT_TIME_MAX):
        """
        Retrieve geographic information
        """
        for row in self._query_geodata(start_time, end_time):
            yield Geo(*row)

    def samples(self, start_time=INSTRUMENT_TIME_MIN, end_time=INSTRUMENT_TIME_MAX):
        """
        Retrieve samples from the dataset.
        """
        for row in self._query_samples(start_time, end_time):
            yield self.Sample(*row)

    def npsamples(self, start_time=INSTRUMENT_TIME_MIN, end_time=INSTRUMENT_TIME_MAX):
        """
        Retrieve samples from the dataset into a NumPy array. Requires NumPy.

        Unlike the “samples” method, which returns a generator, this method
        loads all requested data into memory. Be careful when working with large
        datasets! Rather than loading the entire dataset into memory, things may
        go more smoothly if you work with limited time ranges or a single
        cast/profile at a time.
        """
        try:
            import numpy as np
        except ImportError:
            np = None
            pass

        if not np:
            print("npsamples requires numpy, which could not be imported!")
            return

        # NumPy arrays are fixed-size, so first we need to figure out how many
        # samples we're retrieving.
        sample_count = self._db.execute(
            """select count(*)
                                             from data
                                            where tstamp >= ?
                                              and tstamp < ?""",
            (int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000)),
        ).fetchone()[0]

        # Allocate some space to put the samples...
        samples = np.zeros(
            sample_count,
            dtype={
                "names": self.sample_fields,
                "formats": ["object"] + ["float64"] * (len(self.sample_fields) - 1),
            },
        )

        # ...then load them in.
        i = 0
        for row in self._query_samples(start_time, end_time):
            samples[i] = row
            i += 1

        return samples

    def _query_geodata(self, start_time, end_time):
        return self._query(
            "geodata",
            itertools.chain(("tstamp",), _field_to_column_names(Geo._fields[1:])),
            start_time,
            end_time,
        )

    def _query_samples(self, start_time, end_time):
        return self._query(
            "data",
            itertools.chain(("tstamp",), self.sample_channels),
            start_time,
            end_time,
        )

    def _query(self, table, columns, start_time, end_time):
        if not table in _table_names(self._db):
            return

        column_list = ", ".join(columns)
        start_time = int(start_time.timestamp() * 1000)
        end_time = int(end_time.timestamp() * 1000)

        for row in self._db.execute(
            """select %s
                                         from %s
                                        where tstamp >= ?
                                          and tstamp < ?"""
            % (column_list, table),
            (start_time, end_time),
        ):
            timestamp = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
            yield (timestamp,) + row[1:]

    def profiles(self):
        """
        Retrieve the endpoints of all profiles in the dataset.
        """
        query = """select region.*
                     from region
                    where type = 'PROFILE'"""
        params = ()
        return self._query_regions(query, params)

    def casts(self, direction):
        """
        Retrieve the endpoints of directional casts in the dataset. Argument
        should be one of `pyrsktools.Region.CAST_DOWN`
        or `pyrsktools.Region.CAST_UP`.
        """
        query = """select region.*
                     from regionCast, region
                    where region.regionID = regionCast.regionID
                      and regionCast.type = ?"""
        params = (direction,)
        return self._query_regions(query, params)

    def _query_regions(self, query, params):
        if "region" not in _table_names(self._db):
            return

        cur = self._db.cursor()
        cur.execute(query, params)
        names = {
            name[0]: index
            for name, index in zip(cur.description, range(len(cur.description)))
        }

        for row in self._db.execute(query, params):
            start_time = datetime.fromtimestamp(
                row[names["tstamp1"]] / 1000, tz=timezone.utc
            )
            end_time = datetime.fromtimestamp(
                row[names["tstamp2"]] / 1000, tz=timezone.utc
            )
            label = row[names["label"]] if "label" in names else None
            description = row[names["description"]] if "description" in names else None
            yield Region(self, start_time, end_time, label, description)

        cur.close()

    def close(self):
        self._db.close()


@auto_repr
class Region(object):
    """
    An arbitrary region of time in a dataset.
    """

    CAST_DOWN = "DOWN"
    CAST_UP = "UP"

    def __init__(self, rsk, start_time, end_time, label, description):
        self._rsk = rsk
        self.start_time = start_time
        self.end_time = end_time
        self.label = label
        self.description = description

    def samples(self):
        """
        Retrieve samples from within the region.
        """
        return self._rsk.samples(self.start_time, self.end_time)

    def npsamples(self):
        """
        Retrieve samples from within the region into a Numpy array.
        """
        return self._rsk.npsamples(self.start_time, self.end_time)


def open(rsk):
    """
    Open an RSK file. Returns an RSK object ready for consumption.
    """
    return RSK(rsk)


def _table_names(db):
    """
    Retrieve a list of table names from a database.
    """

    return [
        row[0]
        for row in db.execute(
            "select name from sqlite_master where type = 'table'"
        ).fetchall()
    ]


def _column_names(db, table):
    """
    Retrieve the column names of a database table.
    """

    # Can't use bound values for table names. String substitution it is.
    return [row[1] for row in db.execute("pragma table_info('%s')" % table).fetchall()]


def _field_to_column_names(names):
    """
    Convert field names to database column names.

    The convention of this project is to use snake_case for field names (in the
    belief that this is the most Pythonic approach); conflictingly, the
    convention of the RSK schema is to use camelCase. This function converts
    from snake case to camel case with the intent of being used to help
    auto-generate SQL queries.
    """
    return [re.sub(r"_(.)", lambda m: m.group(1).upper(), name) for name in names]

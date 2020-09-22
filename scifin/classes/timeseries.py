from typing import List, Optional, Iterable
from numbers import Number

from datetime import datetime
import pytz
import pandas as pd


class UnmatchedDataLength(Exception):
    """
    If the lengths of time indices and data disagree
    """
    pass


class UnknownTimeZone(Exception):
    """
    If the input timezone is unknown, i.e. not in pytz library
    """
    pass


# TODO: add methods to fill NaN values, such as ffill/bfill/mean/interpolate
class TimeSeries:
    """
    A time series data class

    (Note: Currently assuming there cannot be any data points
    with empty indices or data value)

    Attributes
    ----------
    series
        Time series data recast as pandas series

    Raises
    ------
    UnmatchedDataLength
        If the lengths of time indices and data disagree
    UnknownTimeZone
        If the input time_zone is not in the list of
        pytz timezones

    """

    def __init__(
            self,
            time_indices: List[datetime],  # TODO: Add separate func to infer or use Time objects instead
            data: list,
            time_zone: str = 'UTC',
            fill_none_method: Optional[str] = None   # TODO
    ):
        """

        Parameters
        ----------
        time_indices
            List of indices of the time series. Must be datetime objects
        data
            List of data value of the time series
        time_zone
            Timezone for datetime objects. Default = 'UTC'
        fill_none_method
            Fill empty data/index points method.
            Allowed methods are ...
            Default = None

        """

        if len(time_indices) != len(data):
            raise UnmatchedDataLength(
                "The lengths of datetime indices and data must be the same"
            )

        if time_zone not in pytz.all_timezones:
            raise UnknownTimeZone(
                "The input timezone must be in the list of timezones in the pytz library"
            )

        self.data = data
        self.times = [
            time.replace(tzinfo=pytz.timezone(time_zone))
            if time is not None else None
            for time in time_indices
        ]
        self.series = pd.Series(data, index=self.times).sort_index()


class NumericalTimeSeries(TimeSeries):
    """
    A TimeSeries with numercial data

    """

    def __init__(
            self,
            time_indices: List[datetime],
            data: List[Number],
            time_zone: str = 'UTC',
            fill_none_method: Optional[str] = None,  # TODO
            unit: Optional[str] = None,
    ):
        super().__init__(time_indices, data, time_zone, fill_none_method)
        self.unit = unit

    @property
    def time_intervals(self):
        """
        Time intervals between neighbouring data points.
        Return an empty list if the length of time series <= 1

        """

        try:
            return [
                times[1] - times[0] for times in zip(self.series.index[:-1], self.series.index[1:])
            ]
        except IndexError:
            return []

    @property
    def differences(self):
        """
        Differences between neighbouring data points.
        Return an empty list if the length of time series <= 1

        """

        try:
            return [
                values[1] - values[0] for values in zip(self.series.values[:-1], self.series.values[1:])
            ]
        except IndexError:
            return []


class CategoricalTimeSeries(TimeSeries):
    """
    A time series data class

    """

    def __init__(
            self,
            time_indices: List[datetime],
            data: List[str],
            time_zone: str = 'UTC',
            fill_none_method: Optional[str] = None  # TODO
    ):
        super().__init__(time_indices, data, time_zone, fill_none_method)

    def value_counts(self, normalize: bool = False) -> pd.Series:
        """
        Return occurrence counts/percentages of each categories

        Parameters
        ----------
        normalize
            If True, return the percentage instead

        """

        return self.series.value_counts(normalize).sort_index()

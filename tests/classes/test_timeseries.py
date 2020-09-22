import pytest
from datetime import datetime, timedelta
import pytz

from pandas.testing import assert_series_equal
import pandas as pd

from scifin.classes.timeseries import (
    TimeSeries, NumericalTimeSeries, CategoricalTimeSeries,
    UnmatchedDataLength, UnknownTimeZone
)


class TestTimeSeries:

    def test_unmatched_lengths_error(self):

        with pytest.raises(UnmatchedDataLength) as exc_info:
            TimeSeries(
                time_indices=[datetime(2020, 1, 1)],
                data=[0, 1]
            )

        expected_error_msg = (
            "The lengths of datetime indices and data must be the same"
        )
        assert exc_info.match(expected_error_msg)

    def test_unknown_timezone_error(self):
        with pytest.raises(UnknownTimeZone) as exc_info:
            TimeSeries(
                time_indices=[datetime(2020, 1, 1)],
                data=[0],
                time_zone='ABC'
            )

        expected_error_msg = (
            "The input timezone must be in the list of timezones in the pytz library"
        )
        assert exc_info.match(expected_error_msg)

    @pytest.mark.parametrize(
        'time_zone',
        [
            'EST',
            'GMT'
        ]
    )
    def test_non_empty_example(
            self, time_zone
    ):

        times = [
            datetime(2020, 1, 1),
            datetime(2020, 2, 1)
        ]
        example = TimeSeries(times, [0, 10], time_zone=time_zone)

        assert example.data == [0, 10]

        expected_times = {
            'EST': [
                datetime(2020, 1, 1, tzinfo=pytz.timezone('EST')),
                datetime(2020, 2, 1, tzinfo=pytz.timezone('EST'))
            ],
            'GMT': [
                datetime(2020, 1, 1, tzinfo=pytz.timezone('GMT')),
                datetime(2020, 2, 1, tzinfo=pytz.timezone('GMT'))
            ],
        }

        assert example.times == expected_times[time_zone]

        assert_series_equal(
            example.series,
            pd.Series([0, 10], index=expected_times[time_zone])
        )


class TestNumericalTimeSeries:

    def test_non_empty_example(self):

        times = [
            datetime(2020, 1, 1),
            datetime(2020, 2, 1)
        ]
        example = NumericalTimeSeries(times, [0, 10])

        assert example.time_intervals == [
            timedelta(days=31)
        ]
        assert example.differences == [10]


class TestCategoricalTimeSeries:

    def test_non_empty_example(self):

        times = [
            datetime(2020, 1, 1),
            datetime(2020, 2, 1)
        ]
        example = CategoricalTimeSeries(times, ['A', 'B'])
        counts = pd.Series(
            [1, 1], index=['A', 'B']
        )
        assert_series_equal(example.value_counts(), counts)

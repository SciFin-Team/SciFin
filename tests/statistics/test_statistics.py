import pytest

from scifin.statistics.distributions import (
    upper_incomplete_gamma
)


class TestUpperIncompleteGamma:

    def test_incorrect_arg_type(self):
        """
        Function that tests the TypeError cases

        """

        with pytest.raises(TypeError) as exc_info:
            upper_incomplete_gamma(a='A', z=0.3)

        expected_error_msg = (
            'type of argument "a" must be one of (int, float); got str instead'
        )
        assert str(exc_info.value) == expected_error_msg

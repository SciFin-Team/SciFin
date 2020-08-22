
<p align="center">
  <img src="https://raw.githubusercontent.com/SciFin-Team/SciFin/master/docs/logos/logo_scifin_github.jpg" width=400 title="hover text">
</p>



# SciFin

SciFin is a python package for Science and Finance.

## Summary

The SciFin package is a Python package designed to gather and develop methods for scientific studies and financial services. It originates from the observation that numerous methods developed in scientific fields (such as mathematics, physics, biology and climate sciences) have direct applicability in finance and that, conversely, multiple methods developed in finance can benefit science.

The development goal of this package is to offer a toolbox that can be used both in research and business. Its purpose is not only to bring these fields together, but also to increase interoperability between them, helping science turn into business and finance to get new insights from science. Some functions are thus neutral to any scientific or economical fields, while others are more specialized to precise tasks. The motivation behind this design is to provide tools that perform advanced tasks while remaining simple (not depending on too many parameters).


## Table of Contents

- **[Development Stage](#development-stage)**<br>
- **[Installation](#installation)**<br>
- **[Usage](#usage)**<br>
- **[Contributing](#contributing)**<br>
- **[Credits](#credits)**<br>
- **[License](#license)**<br>
- **[Contacts](#contacts)**<br>


## Development Stage

The current development is focused on the following topics:

| Subpackage | Short Description | Development Stage |
| :-----: | :-----: | :-----: |
| [`classifier`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/classifier) | classification techniques | ■ □ □ □ □ |
| [`fouriertrf`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/fouriertrf) | Fourier transforms | ■ □ □ □ □ |
| [`geneticalg`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/geneticalg) | genetic algorithms | ■ ■ ■ □ □ |
| [`marketdata`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/marketdata) | reading market data | ■ □ □ □ □ |
| [`montecarlo`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/montecarlo) | Monte Carlo simulations | ■ □ □ □ □ |
| [`neuralnets`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/neuralnets) | neural networks | □ □ □ □ □ |
| [`statistics`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/statistics) | basic statistics | ■ □ □ □ □ |
| [`timeseries`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/timeseries) | time series analysis | ■ ■ ■ ■ □ |

The topics already developed are time series analysis, genetic algorithms and statistics.

A lot of development still needs to be done. Other topics will also later follow.


## Installation

Installing SciFin on Linux or Mac is very easy, you can simply run this on a terminal:  
`pip install SciFin`  

You can also access the last version of the package [on PyPI](https://pypi.org/project/scifin/).

If you encounter problems during installation or after and think you know how the problem can be improved, please share it with me.

Version 0.0.8 may lead to a small problem from pandas. If you get an error message such as:  
`ImportError: cannot import name 'urlencode' from 'pandas.io.common'`  
it is advised to install pandas version 1.0.3 using e.g. the command line:  
`pip install pandas==1.0.3`.


## Usage

The code is growing fast and many classes and function acquire new features. Hence, one version can be significantly different from the previous one at the moment. That's what makes development exciting! But that can also be confusing.

A documentation of the code should help users. Once ready, this documentation will start appearing on [SciFin's Wiki page](https://github.com/SciFin-Team/SciFin/wiki).

If you encounter any problem while using SciFin, please do not hesitate to report it to us by [creating an issue](https://docs.github.com/en/github/managing-your-work-on-github/creating-an-issue).


## Contributing

The package tries to follow the style guide for Python code [PEP8](https://www.python.org/dev/peps/pep-0008/). If you find any part of the code unclear or departing from this style, please let me know. As for docstrings, the format we try to follow here is given by the [numpy doc style](https://numpydoc.readthedocs.io/en/latest/format.html).

It is strongly advised to have a fair knowledge of Python to contribute, at least a strong motivation to learn, and recommanded to read the following [Python3 Tutorial](https://www.python-course.eu/python3_course.php) before joining the project.

To know more about the (evolving) rules that make the project self-consistent and eases interaction between contributors, please refer to details in the [Contributing](https://github.com/SciFin-Team/SciFin/blob/master/CONTRIBUTING.md) file.


## Credits

All the development up to now has been done by Fabien Nugier. New contributors will join soon.


## License

SciFin is currently developed under the MIT license.

Please keep in mind that SciFin and its developers hold no responsibility for any wrong usage or losses related to the package usage.

For more details, please refer to the [license](https://github.com/SciFin-Team/SciFin/blob/master/LICENSE).


## Contacts

If you have comments or suggestions, please reach Fabien Nugier. Thank you very much in advance for your feedback.




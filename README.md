
<p align="center">
  <img src="https://raw.githubusercontent.com/SciFin-Team/SciFin/master/docs/logos/logo_scifin_github.jpg" width=400 title="hover text">
</p>



# SciFin

SciFin is a python package for Science and Finance.

## Summary

The SciFin package is a Python package designed to gather and develop methods for scientific studies and financial services. It originates from the observation that numerous methods developed in scientific fields (such as mathematics, physics, biology and climate sciences) have direct applicability in finance and that, conversely, multiple methods developed in finance can benefit science.

The development goal of this package is to offer a toolbox that can be used to derive specific applications both in research and business. Its purpose is not only to bring these fields together, but also to increase interoperability between them, helping science turn into business and finance to get new insights from science. Some functions will thus be neutral to any scientific or economic fields, while others will be more specialized for precise tasks. The motivation behind this design is to provide tools that perform advanced tasks without depending on too many parameters.


### Table of Contents
**[Development Stage](#development-stage)**<br>
**[Installation](#installation)**<br>
**[Usage](#usage)**<br>
**[Contributing](#contributing)**<br>
**[Credits](#credits)**<br>
**[License](#license)**<br>
**[Contacts](#contacts)**<br>


## Development Stage

The current development is focused on the following topics:

| Subpackage | Short Description | Development Stage |
| :-----: | :-----: | :-----: |
| `classifier` | classification techniques | ■ □ □ □ □ |
| `fouriertrf` | Fourier transforms | ■ □ □ □ □ |
| `geneticalg` | genetic algorithms | ■ ■ ■ □ □ |
| `marketdata` | reading market data | ■ □ □ □ □ |
| `montecarlo` | Monte Carlo simulations | □ □ □ □ □ |
| `neuralnets` | neural networks | □ □ □ □ □ |
| `statistics` | basic statistics | ■ □ □ □ □ |
| `timeseries` | time series analysis | ■ ■ ■ ■ □ |

The topics already developed are time series analysis, genetic algorithms and statistics. A lot of development still needs to be done. Other topics will also later follow.


## Installation

Installing SciFin on Linux or Mac is very easy, you can simply run this on a terminal:  
`pip install SciFin`  
You can also access the last version of the package on PyPI by clicking [--> Here <--](https://pypi.org/project/scifin/).

If you encounter problems during installation or after and think you know how the problem can be improved, please share it with me.

Version 0.0.8 may lead to a small problem from pandas. If you get an error message such as:  
`ImportError: cannot import name 'urlencode' from 'pandas.io.common'`  
it is advised to install pandas version 1.0.3 using e.g. the command line  
`pip install pandas==1.0.3`.


## Usage

To be updated.


## Contributing

The package written tries to follow the style guide for Python code [PEP8](https://www.python.org/dev/peps/pep-0008/). If you find any part of the code unclear, please let me know. As for docstrings, the format we try to follow here is given by the [numpy doc style](https://numpydoc.readthedocs.io/en/latest/format.html).

It is strongly advised to have a fair knowledge of Python, or at least a strong motivation to learn, and recommanded to read the following [Python3 Tutorial](https://www.python-course.eu/python3_course.php) before contributing to the project.


## Credits

All the development up to now has been done by Fabien Nugier.


## License

Please keep in mind that SciFin and its developers hold no responsibility for any wrong usage or losses related to the package usage.

For more details, please refer to the [license](https://github.com/SciFin-Team/SciFin/blob/master/LICENSE).


## Contacts

If you have comments or suggestions, please reach Fabien Nugier. Thank you very much in advance for your feedback.




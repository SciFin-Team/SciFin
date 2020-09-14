
<p align="center">
  <img src="https://raw.githubusercontent.com/SciFin-Team/SciFin/master/docs/logos/logo_scifin_github.jpg" width=400 title="hover text">
</p>



# SciFin

SciFin is a python package for Science and Finance.


## Summary

The SciFin package is a Python library designed to gather and develop methods for scientific studies
and financial services. It originates from the observation that numerous methods developed in scientific
fields (such as mathematics, physics, biology, climate sciences, medicine) have direct applicability in finance
and that, conversely, multiple methods developed in finance can benefit science.

The development goal of this package is to offer a toolbox that can be used both in research and business analyses.
Its purpose is not only to bring these fields together, but also to increase interoperability between them,
helping science turn into sound business and finance get new insights from science. Core functions of SciFin
thus try to remain neutral to any scientific or economical context, while derived methods are specializing towards
specific applications.

SciFin's toolbox thus intend to provide functions that perform advanced tasks while remaining simple,
i.e. depending on a minimal amount of parameters. Doing so can increase the scope of users while focusing
on the real nature of mathematical objects to solve problems, leaving most of the specialization to the user him/herself.


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
| [`classifier`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/classifier) | classification techniques | ■ ■ □ □ □ |
| [`fouriertrf`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/fouriertrf) | Fourier transforms | ■ □ □ □ □ |
| [`geneticalg`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/geneticalg) | genetic algorithms | ■ ■ ■ □ □ |
| [`marketdata`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/marketdata) | reading market data | ■ ■ □ □ □ |
| [`montecarlo`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/montecarlo) | Monte Carlo simulations | ■ □ □ □ □ |
| [`neuralnets`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/neuralnets) | neural networks | □ □ □ □ □ |
| [`statistics`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/statistics) | basic statistics | ■ ■ □ □ □ |
| [`timeseries`](https://github.com/SciFin-Team/SciFin/tree/master/scifin/timeseries) | time series analysis | ■ ■ ■ ■ □ |

Topics already well developed are time series analysis and genetic algorithms. Topics recently addressed are statistics
and clustering/classification methods.

A lot of development still needs to be done. Other topics will later follow.


## Installation

Installing SciFin on Linux or Mac is very easy, you can simply run this command on a terminal:  
`pip install SciFin`  

You can also access the last version of the package [on PyPI](https://pypi.org/project/scifin/) and download it from there.

If you encounter problems during installation or after and think you know how the problem can be improved,
please share it with us.

**It is advised to install version 0.1.0 or later.**

Version 0.0.8 may lead to a small problem from pandas. If you get an error message such as:  
`ImportError: cannot import name 'urlencode' from 'pandas.io.common'`  
it is advised to install pandas version 1.0.3 using e.g. the command line:  
`pip install pandas==1.0.3`.


## Usage

The code is growing fast and many classes and functions acquire new features. Hence, at the moment, one version can be
significantly different from its previous one. That's what makes development exciting! But that can also be confusing.

A documentation of the code should help users. Once ready, this documentation will start appearing
on [SciFin's Wiki page](https://github.com/SciFin-Team/SciFin/wiki).

If you encounter any problem while using SciFin, please do not hesitate to report it to us
by [creating an issue](https://docs.github.com/en/github/managing-your-work-on-github/creating-an-issue).


## Contributing

The package tries to follow the style guide for Python code [PEP8](https://www.python.org/dev/peps/pep-0008/).
If you find any part of the code unclear or departing from this style, please let me know. As for docstrings,
the format we try to follow here is given by the [numpy doc style](https://numpydoc.readthedocs.io/en/latest/format.html).

It is strongly advised to have a fair knowledge of Python to contribute, at least a strong motivation to learn,
and recommended to read the following [Python3 Tutorial](https://www.python-course.eu/python3_course.php)
before joining the project.

To know more about the (evolving) rules that make the project self-consistent and eases interaction between contributors,
please refer to details in the [Contributing](https://github.com/SciFin-Team/SciFin/blob/master/CONTRIBUTING.md) file.


## Credits

All the development was done by Fabien Nugier until recently, new contributors have now joined the project.


## License

SciFin is currently developed under the MIT license.

Please keep in mind that SciFin and its developers hold no responsibility of any king for any wrong usage of its
content, or any losses related to the package usage.

For more details, please refer to the [license](https://github.com/SciFin-Team/SciFin/blob/master/LICENSE).


## Contacts

If you have comments or suggestions, please reach Fabien Nugier.
Thank you very much in advance for your feedback.


## References

SciFin uses code from scientific papers, books and online articles sometimes.
Here are presented lists of materials which have been used to develop some functionalities in the code:

**Books:**
- [_Analysis of Financial Time Series_, Ruey S. Tsay](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354), Wiley (2010).
- [_Machine Learning for Asset Managers_, Marcos López de Prado](https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545), Cambridge University Press (2018).
- [_Advances in Financial Machine Learning_, Marcos López de Prado](https://www.wiley.com/en-fr/Advances+in+Financial+Machine+Learning-p-9781119482086), Wiley (2020).  
  
**Articles:**
- [_Continuous Genetic Algorithm From Scratch With Python_, Cahit bartu yazıcı](https://towardsdatascience.com/continuous-genetic-algorithm-from-scratch-with-python-ff29deedd099).  
  
**Websites:**
- [Wikipedia](https://www.wikipedia.org/)

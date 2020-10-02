import pytest
import numpy as np

from scifin.statistics import distributions as dis
from scifin.statistics import statistics


class TestStandardNormalFunctions:
    """
    Tests the standard normal functions.
    """

    def test_pdf(self):
        assert dis.standard_normal_pdf(0.) == pytest.approx(0.3989422804014327, 1e-12)
        assert dis.standard_normal_pdf(1.) == pytest.approx(0.24197072451914337, 1e-12)
        assert dis.standard_normal_pdf(1.) == dis.standard_normal_pdf(-1.)

    def test_cdf(self):
        assert dis.standard_normal_cdf(0.) == 0.5
        assert dis.standard_normal_cdf(1.6448536269514722) == pytest.approx(0.95, 1e-12)
        assert dis.standard_normal_cdf(1.) == 1 - dis.standard_normal_cdf(-1.)

    def test_quantile(self):
        assert dis.standard_normal_quantile(0.95) == pytest.approx(1.6448536269514722, 1e-12)
        assert dis.standard_normal_quantile(0.99) == pytest.approx(2.3263478740408408, 1e-12)
        assert dis.standard_normal_quantile(0.95) + dis.standard_normal_quantile(0.05) == pytest.approx(0.)
        assert dis.standard_normal_quantile(0.99) + dis.standard_normal_quantile(0.01) == pytest.approx(0.)


# CONTINUOUS DISTRIBUTIONS

class TestNormal:
    """
    Tests the class Normal.
    """
    
    def test_zero_sigma_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Normal(mu=1., sigma=0., name="MyGaussian")

    def test_attributes(self):
        d1 = dis.Normal(mu=1., sigma=0.3, name="MyGaussian")

        assert d1.type == 'Normal'
        assert d1.support == 'R'
        assert d1.mu == 1.
        assert d1.sigma == 0.3
        assert d1.mean == 1.
        assert d1.variance == 0.09
        assert d1.std == 0.3
        assert d1.skewness == 0.
        assert d1.kurtosis == 3.
        assert d1.median == 1.
        assert d1.mode == 1.
        assert d1.MAD == pytest.approx(0.2393653682408596, 1e-12)
        assert d1.entropy == pytest.approx(0.2149657288787366, 1e-12)
        assert d1.name == "MyGaussian"

    def test_methods(self):
        d1 = dis.Normal(mu=1., sigma=0.3, name="MyGaussian")

        assert list(d1.pdf([1., 2.])) == [1.329807601338109, 0.005140929987637022]
        assert list(d1.cdf([1., 2.])) == [0.5, 0.9995709396668031]
        assert d1.quantile(p=0.95) == pytest.approx(1.4934560880854417, 1e-12)
        assert d1.var(p=0.1)  == pytest.approx(0.61553453033662, 1e-12)
        assert d1.cvar(p=0.1) == pytest.approx(1.5264949957974605, 1e-12)


class TestUniform:

    """
    Tests the class Uniform.
    """

    def set_b_greater_than_a_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Uniform(a=2., b=1.)

    def test_attributes(self):
        d1 = dis.Uniform(a=-1., b=2., name="MyUniform")

        assert d1.type == 'Uniform'
        assert d1.support == '[a,b]'

        assert d1.a == -1.
        assert d1.b == 2.

        assert d1.mean == 0.5
        assert d1.variance == 0.75
        assert d1.std == pytest.approx(0.8660254037844386, 1e-12)
        assert d1.skewness == 0.
        assert d1.kurtosis == 1.8
        assert d1.median == 0.5
        assert d1.mode == 'Any value between a and b.'
        assert d1.entropy == pytest.approx(1.0986122886681098, 1e-12)
        assert d1.name == "MyUniform"

    def test_methods(self):
        d1 = dis.Uniform(a=-1., b=2., name="MyUniform")
    
        assert list(d1.pdf([0, 1])) == [0.3333333333333333, 0.3333333333333333]
        assert list(d1.cdf([0, 1])) == [0.3333333333333333, 0.6666666666666666]


class TestWeibull:
    """
    Tests the class Weibull.
    """

    def test_negative_kappa_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Weibull(k=-1, lmbda=1)

    def test_negative_lambda_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Weibull(k=1, lmbda=-1)

    def test_attributes(self):
        d1 = dis.Weibull(k=0.5, lmbda=1., name="MyWeibull")

        assert d1.type == 'Weibull'
        assert d1.support == 'R+'

        assert d1.k == 0.5
        assert d1.lmbda == 1.

        assert d1.mean == 2.0
        assert d1.variance == 20.0
        assert d1.std == pytest.approx(4.47213595499958, 1e-12)
        assert d1.skewness == pytest.approx(6.6187612133993765, 1e-12)
        assert d1.kurtosis == pytest.approx(87.71999999999998, 1e-12)
        assert d1.median == pytest.approx(0.4804530139182014, 1e-12)
        assert d1.mode == 0
        assert d1.entropy == pytest.approx(1.1159315156584124, 1e-12)
        assert d1.name == "MyWeibull"

    def test_methods(self):
        d1 = dis.Weibull(k=0.5, lmbda=1., name="MyWeibull")

        assert list(d1.pdf([1, 2])) == [0.18393972058572117, 0.08595474576918094]
        assert list(d1.cdf([1, 2])) == [0.6321205588285577, 0.7568832655657858]
        assert d1.var(p=0.1) == pytest.approx(0.011100838259683056, 1e-12)
        assert d1.cvar(p=0.1) == pytest.approx(2.2218218695753356, 1e-12)


class Rayleigh:
    """
    Tests the class Rayleigh.
    """

    def test_negative_sigma_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Rayleigh(sigma=-1)

    def test_attributes(self):
        d1 = dis.Rayleigh(sigma=1., name="MyRayleigh")

        assert d1.type == 'Rayleigh'
        assert d1.support == 'R+'

        assert d1.sigma == 1.

        assert d1.mean == pytest.approx(1.2533141373155001, 1e-12)
        assert d1.variance == pytest.approx(0.42920367320510344, 1e-12)
        assert d1.std == pytest.approx(0.6551363775620336, 1e-12)
        assert d1.skewness == pytest.approx(0.6311106578189364, 1e-12)
        assert d1.kurtosis == pytest.approx(3.245089300687639, 1e-12)
        assert d1.median == pytest.approx(1.1774100225154747, 1e-12)
        assert d1.mode == 1
        assert d1.entropy == pytest.approx(0.9420342421707937, 1e-12)
        assert d1.name == "MyRayleigh"

    def test_methods(self):
        d1 = dis.Rayleigh(sigma=1., name="MyRayleigh")

        assert list(d1.pdf([1, 2])) == [0.6065306597126334, 0.2706705664732254]
        assert list(d1.cdf([1, 2])) == [0.3934693402873666, 0.8646647167633873]


class Exponential:
    """
    Tests the class Exponential.
    """
    def setUp(self):
        # Test AssertionError
        with self.assertRaises(AssertionError):
            dis.Exponential(lmbda=-1)

    def test_attributes(self):
        d1 = dis.Exponential(lmbda=2., name="MyExponential")

        assert d1.type == 'Exponential'
        assert d1.support == 'R+'

        assert d1.lmbda == 2.

        assert d1.mean == 0.5
        assert d1.variance == 0.25
        assert d1.std == 0.5
        assert d1.skewness == 2
        assert d1.kurtosis == 9
        assert d1.median == pytest.approx(0.34657359027997264, 1e-12)
        assert d1.mode == 0
        assert d1.entropy == pytest.approx(0.3068528194400547, 1e-12)
        assert d1.name == "MyExponential"

    def test_methods(self):
        d1 = dis.Exponential(lmbda=2., name="MyExponential")

        assert list(d1.pdf([0, 1, 2])) == [2.0, 0.2706705664732254, 0.03663127777746836]
        assert list(d1.cdf([0, 1, 2])) == [0.0, 0.8646647167633873, 0.9816843611112658]
        assert d1.var(p=0.1) == pytest.approx(0.05268025782891314, 1e-12)
        assert d1.cvar(p=0.1) == pytest.approx(0.5526802578289132, 1e-12)

class Gumbel:
    """
    Tests the class Gumbel.
    """

    def test_negative_beta_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Gumbel(mu=1., beta=-1.)
    
    def test_attributes(self):
        d1 = dis.Gumbel(mu=1., beta=2., name="MyGumbel")

        assert d1.type == 'Gumbel'
        assert d1.support == 'R'
    
        assert d1.mu == 1.
        assert d1.beta == 2.
    
        assert d1.mean == pytest.approx(2.1544313298030655, 1e-12)
        assert d1.variance == pytest.approx(6.579736267392906, 1e-12)
        assert d1.std == pytest.approx(2.565099660323728, 1e-12)
        assert d1.skewness == pytest.approx(1.1395470994046488, 1e-12)
        assert d1.kurtosis == 5.4
        assert d1.median == pytest.approx(1.7330258411633288, 1e-12)
        assert d1.mode == 1
        assert d1.entropy == pytest.approx(2.270362845461478, 1e-12)
        assert d1.name == "MyGumbel"
    
    def test_methods(self):
        d1 = dis.Gumbel(mu=1., beta=2., name="MyGumbel")

        assert list(d1.pdf([0., 5., 10.])) == [0.15852096053897108, 0.05910247579657157, 0.005493134841201401]
        assert list(d1.cdf([0., 5., 10.])) == [0.1922956455479649, 0.8734230184931167, 0.9889524805037951]
        assert d1.var(p=0.1) == -0.6680648904959114

class Laplace:
    """
    Tests the class Laplace.
    """

    def test_negative_b_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Laplace(mu=0, b=-1)

    def test_attributes(self):
        d1 = dis.Laplace(mu=0, b=1, name="MyLaplace")

        assert d1.type == 'Laplace'
        assert d1.support == 'R'
    
        assert d1.mu == 0
        assert d1.b == 1
    
        assert d1.mean == 0
        assert d1.variance == 2
        assert d1.std == pytest.approx(1.4142135623730951, 1e-12)
        assert d1.skewness == 0
        assert d1.kurtosis == 6
        assert d1.median == 0
        assert d1.mode == 0
        assert d1.entropy == pytest.approx(1.6931471805599452, 1e-12)
        assert d1.name == "MyLaplace"
    
    def test_methods(self):
        d1 = dis.Laplace(mu=0, b=1, name="MyLaplace")

        assert list(d1.pdf([0, 1, 2])) == [0.5, 0.18393972058572117, 0.06766764161830635]
        assert list(d1.cdf([0, 1, 2])) == [0.5, 0.8160602794142788, 0.9323323583816936]
        assert d1.var(p=0.1) == pytest.approx(-1.6094379124341003, 1e-12)
        assert d1.cvar(p=0.1) == pytest.approx(0.28993754582601117, 1e-12)

class Levy:
    """
    Tests the class Levy.
    """

    def test_negative_c_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Levy(mu=0, c=-1)

    def test_attributes(self):
        d1 = dis.Levy(mu=0, c=1, name="MyLevy")

        assert d1.type == 'Levy'
        assert d1.support == '[mu == Infinity'

        assert d1.mu == 0
        assert d1.c == 1

        assert d1.mean == 'Infinity'
        assert d1.variance == 'Infinity'
        assert d1.std == 'Infinity'
        assert d1.skewness is None
        assert d1.kurtosis is None
        assert d1.median == pytest.approx(0.11373410577989317, 1e-12)
        assert d1.mode == pytest.approx(0.3333333333333333, 1e-12)
        assert d1.entropy == pytest.approx(6.64896560279378, 1e-12)
        assert d1.name == "MyLevy"

    def test_methods(self):
        d1 = dis.Levy(mu=0, c=1, name="MyLevy")

        assert list(d1.pdf([0, 1, 2])) == [0, 0.24197072451914337, 0.10984782236693061]
        assert list(d1.cdf([0, 1, 2])) == [0, 0.31731050786291415, 0.4795001221869535]

class Cauchy:
    """
    Tests the class Cauchy.
    """

    def test_negative_b_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Cauchy(a=0, b=-1)

    def test_attributes(self):
        d1 = dis.Cauchy(a=0, b=1, name="MyCauchy")

        assert d1.type == 'Cauchy'
        assert d1.support == 'R'

        assert d1.a == 0
        assert d1.b == 1

        assert d1.mean is None
        assert d1.variance is None
        assert d1.std is None
        assert d1.skewness is None
        assert d1.kurtosis is None
        assert d1.median == 0
        assert d1.mode == 0
        assert d1.entropy == pytest.approx(2.5310242469692907, 1e-12)
        assert d1.name == "MyCauchy"

    def test_methods(self):
        d1 = dis.Cauchy(a=0, b=1, name="MyCauchy")

        assert list(d1.pdf([0, 1, 2])) == [0.3183098861837907, 0.15915494309189535, 0.06366197723675814]
        assert list(d1.cdf([0, 1, 2])) == [0.5, 0.75, 0.8524163823495667]

# DISCRETE DISTRIBUTIONS

class Poisson:
    """
    Tests the class Poisson.
    """

    def test_negative_lambda_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Poisson(lmbda=-1., name="MyPoisson")

    def test_attributes(self):
        d1 = dis.Poisson(lmbda=0.5, name="MyPoisson")

        assert d1.type == 'Poisson'
        assert d1.support == 'N'
        assert d1.lmbda == 0.5
        assert d1.mean == 0.5
        assert d1.variance == 0.5
        assert d1.std == pytest.approx(0.7071067811865476, 1e-12)
        assert d1.skewness == pytest.approx(1.414213562373095, 1e-12)
        assert d1.kurtosis == pytest.approx(4.414213562373095, 1e-12)
        assert d1.median == 0.
        assert d1.k_max == 1000
        assert d1.mode == 0.
        assert d1.entropy == pytest.approx(0.8465735902799727, 1e-12)
        assert d1.name == "MyPoisson"

    def test_methods(self):
        d1 = dis.Poisson(lmbda=0.5, name="MyPoisson")

        assert list(d1.pmf([1, 2])) == [0.3032653298563167, 0.07581633246407918]
        assert list(d1.cdf([1, 2])) == [0.9097959895689501, 0.9856123220330293]


class Binomial:
    """
    Tests the class Binomial.
    """

    def test_negative_n_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Binomial(n=-10, p=0.5)

    def test_float_n_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Binomial(n=-1., p=0.5)

    def test_negative_p_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Binomial(n=10, p=-0.5)

    def test_p_greater_than_1_raises_error(self):
        with pytest.raises(AssertionError):
            dis.Binomial(n=10, p=-0.5)

    def test_attributes(self):
        d1 = dis.Binomial(n=10, p=0.7, name="MyBinomial")
        
        assert d1.type == 'Binomial'
        assert d1.support == '{0,1,...,n}'

        assert d1.n == 10
        assert d1.p == 0.7
        assert d1.q == 0.3

        assert d1.mean == 7.0
        assert d1.variance == pytest.approx(2.1000000000000005, 1e-12)
        assert d1.std == pytest.approx(1.449137674618944, 1e-12)
        assert d1.skewness == pytest.approx(-0.27602622373694163, 1e-12)
        assert d1.kurtosis == pytest.approx(2.876190476190476, 1e-12)
        assert d1.median == 7.0

        assert d1.mode == 7.0
        assert d1.entropy == pytest.approx(2.58229024912634, 1e-12)
        assert d1.name == "MyBinomial"

    def test_methods(self):
        d1 = dis.Binomial(n=10, p=0.7, name="MyBinomial")

        assert list(d1.pmf([1, 2])) == [0.00013778100000000018, 0.0014467005000000015]
        assert list(d1.cdf([1, 2])) == [0.00014368590000000018, 0.0015903864000000017]


class TestMarcenkoPastur:

    def test_refactoring(self):
        X = np.array([1.91315248, 1.81189185, 1.72542052, 1.65689567, 1.64134677,
                      1.62151478, 1.58697418, 1.53606406, 1.51293438, 1.47587683,
                      1.47113945, 1.43413408, 1.42520198, 1.40412494, 1.3886599,
                      1.37391991, 1.35689981, 1.34236907, 1.31795082, 1.30957462,
                      1.29085016, 1.27863071, 1.26960764, 1.25244721, 1.24404906,
                      1.23392868, 1.21928507, 1.20603135, 1.19923494, 1.1825206,
                      1.16721998, 1.16247636, 1.13830932, 1.13227976, 1.11265137,
                      1.10132257, 1.09624903, 1.07900696, 1.06956545, 1.06595237,
                      1.04785689, 1.04412674, 1.02471149, 1.02143057, 1.01571143,
                      0.993282, 0.98717485, 0.97478547, 0.96958049, 0.9559686,
                      0.95172736, 0.94866285, 0.92993663, 0.92236546, 0.91347391,
                      0.90025847, 0.88726593, 0.88652546, 0.87130921, 0.86891738,
                      0.85601713, 0.84471017, 0.84039224, 0.82083072, 0.81597161,
                      0.80310027, 0.79844503, 0.78560994, 0.78167253, 0.77388754,
                      0.76554336, 0.7586511, 0.7545136, 0.7437965, 0.7401452,
                      0.73424185, 0.71591487, 0.70597488, 0.69454806, 0.69074335,
                      0.68826198, 0.66965944, 0.65086124, 0.64712531, 0.6396971,
                      0.62836684, 0.62156755, 0.61095141, 0.60050098, 0.57893671,
                      0.57559399, 0.56958534, 0.56116821, 0.54794375, 0.53189713,
                      0.52770162, 0.51966797, 0.51144757, 0.49444351, 0.47717652])
        X2 = np.copy(X)
        n_features0 = 100
        n_obs0 = 10

        _, sigma1, _ = statistics.marcenko_pastur_fit_params(n_features=n_features0,
                                                             n_obs=n_obs0,
                                                             sigma_ini=np.sqrt(0.5),
                                                             e_val=np.diag(X),
                                                             bwidth=0.01)

        print(sigma1)

        mp_dist = dis.MarcenkoPastur(0.5, 0.1)
        #X2 = np.diag(X2)
        print(X2.shape)
        # Should n_pts be the same as n_features??
        mp_dist.fit(X2, n_pts=100, bandwidth=0.01, initial_params=[np.sqrt(0.5), 0.1])

        print(mp_dist.sigma)
        print(mp_dist.lambda_)
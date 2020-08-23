# Solving relative path problem
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

# Import Unittest
import unittest

# Import my package
from scifin.statistics import distributions as dis
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#


class TestStandardNormalFunctions(unittest.TestCase):
    """
    Tests the standard normal functions.
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_pdf(self):
        self.assertEqual(dis.standard_normal_pdf(0.), 0.3989422804014327)
        self.assertEqual(dis.standard_normal_pdf(1.), 0.24197072451914337)
        self.assertEqual(dis.standard_normal_pdf(1.), dis.standard_normal_pdf(-1.))

    def test_cdf(self):
        self.assertEqual(dis.standard_normal_cdf(0.), 0.5)
        self.assertEqual(dis.standard_normal_cdf(1.6448536269514722), 0.95)
        self.assertEqual(dis.standard_normal_cdf(1.), 1-dis.standard_normal_cdf(-1.))

    def test_quantile(self):
        self.assertEqual(dis.standard_normal_quantile(0.95), 1.6448536269514722)
        self.assertEqual(dis.standard_normal_quantile(0.99), 2.3263478740408408)
        self.assertAlmostEqual(dis.standard_normal_quantile(0.95) + dis.standard_normal_quantile(0.05), 0.)
        self.assertAlmostEqual(dis.standard_normal_quantile(0.99) + dis.standard_normal_quantile(0.01), 0.)
        
        

# CONTINUOUS DISTRIBUTIONS
        
class TestNormal(unittest.TestCase):
    """
    Tests the class Normal.
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        
        # Test AssertionError
        with self.assertRaises(AssertionError):
            dis.Normal(mu=1., sigma=0., name="MyGaussian")
        
        # For the later tests
        self.d1 = dis.Normal(mu=1., sigma=0.3, name="MyGaussian")
        
    def tearDown(self):
        pass
    
    def test_attributes(self):
        
        self.assertEqual(self.d1.type, 'Normal')
        self.assertEqual(self.d1.support, 'R')
        self.assertEqual(self.d1.mu, 1.)
        self.assertEqual(self.d1.sigma, 0.3)
        self.assertEqual(self.d1.mean, 1.)
        self.assertEqual(self.d1.variance, 0.09)
        self.assertEqual(self.d1.std, 0.3)
        self.assertEqual(self.d1.skewness, 0.)
        self.assertEqual(self.d1.kurtosis, 3.)
        self.assertEqual(self.d1.median, 1.)
        self.assertEqual(self.d1.mode, 1.)
        self.assertEqual(self.d1.MAD, 0.2393653682408596)
        self.assertEqual(self.d1.entropy, 0.2149657288787366)
        self.assertEqual(self.d1.name, "MyGaussian")
        
    def test_methods(self):
        
        self.assertListEqual(list(self.d1.pdf([1.,2.])), [1.329807601338109, 0.005140929987637022])
        self.assertListEqual(list(self.d1.cdf([1.,2.])), [0.5, 0.9995709396668031])
        self.assertEqual(self.d1.quantile(p=0.95), 1.4934560880854417)
        self.assertEqual(self.d1.var(p=0.1), 0.61553453033662)
        self.assertEqual(self.d1.cvar(p=0.1), 1.5264949957974605)
    

    
class Uniform(unittest.TestCase):
    """
    Tests the class Uniform.
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        
        # Test AssertionError
        with self.assertRaises(AssertionError):
            dis.Uniform(a=2., b=1.)
        
        # For the later tests
        self.d1 = dis.Uniform(a=-1., b=2., name="MyUniform")
        
    def tearDown(self):
        pass
    
    def test_attributes(self):
        
        self.assertEqual(self.d1.type, 'Uniform')
        self.assertEqual(self.d1.support, '[a,b]')
        
        self.assertEqual(self.d1.a, -1.)
        self.assertEqual(self.d1.b, 2.)
        
        self.assertEqual(self.d1.mean, 0.5)
        self.assertEqual(self.d1.variance, 0.75)
        self.assertEqual(self.d1.std, 0.8660254037844386)
        self.assertEqual(self.d1.skewness, 0.)
        self.assertEqual(self.d1.kurtosis, 1.8)
        self.assertEqual(self.d1.median, 0.5)
        self.assertEqual(self.d1.mode, 'Any value between a and b.')
        self.assertEqual(self.d1.entropy, 1.0986122886681098)
        self.assertEqual(self.d1.name, "MyUniform")
    
    def test_methods(self):
        
        self.assertListEqual(list(self.d1.pdf([0,1])), [0.3333333333333333, 0.3333333333333333])
        self.assertListEqual(list(self.d1.cdf([0,1])), [0.3333333333333333, 0.6666666666666666])

        
        
class Weibull(unittest.TestCase):
    """
    Tests the class Weibull.
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        
        # Test AssertionError
        with self.assertRaises(AssertionError):
            dis.Weibull(k=-1, lmbda=1)
        with self.assertRaises(AssertionError):
            dis.Weibull(k=1, lmbda=-1)
        
        # For the later tests
        self.d1 = dis.Weibull(k=0.5, lmbda=1., name="MyWeibull")
        
    def tearDown(self):
        pass
    
    def test_attributes(self):
        
        self.assertEqual(self.d1.type, 'Weibull')
        self.assertEqual(self.d1.support, 'R+')
        
        self.assertEqual(self.d1.k, 0.5)
        self.assertEqual(self.d1.lmbda, 1.)
        
        self.assertEqual(self.d1.mean, 2.0)
        self.assertEqual(self.d1.variance, 20.0)
        self.assertEqual(self.d1.std, 4.47213595499958)
        self.assertEqual(self.d1.skewness, 6.6187612133993765)
        self.assertEqual(self.d1.kurtosis, 87.71999999999998)
        self.assertEqual(self.d1.median, 0.4804530139182014)
        self.assertEqual(self.d1.mode, 0)
        self.assertEqual(self.d1.entropy, 1.1159315156584124)
        self.assertEqual(self.d1.name, "MyWeibull")
    
    def test_methods(self):
        
        self.assertListEqual(list(self.d1.pdf([1,2])), [0.18393972058572117, 0.08595474576918094])
        self.assertListEqual(list(self.d1.cdf([1,2])), [0.6321205588285577, 0.7568832655657858])
        self.assertEqual(d10.var(p=0.1)), 0.011100838259683056)
        self.assertEqual(d10.cvar(p=0.1)), 2.2218218695753356)
        


class Rayleigh(unittest.TestCase):
    """
    Tests the class Rayleigh.
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        
        # Test AssertionError
        with self.assertRaises(AssertionError):
            dis.Rayleigh(sigma=-1)
        
        # For the later tests
        self.d1 = dis.Weibull(sigma=1., name="MyRayleigh")
        
    def tearDown(self):
        pass
    
    def test_attributes(self):
        
        self.assertEqual(self.d1.type, 'Rayleigh')
        self.assertEqual(self.d1.support, 'R+')
        
        self.assertEqual(self.d1.sigma, 1.)
        
        self.assertEqual(self.d1.mean, 1.2533141373155001)
        self.assertEqual(self.d1.variance, 0.42920367320510344)
        self.assertEqual(self.d1.std, 0.6551363775620336)
        self.assertEqual(self.d1.skewness, 0.6311106578189364)
        self.assertEqual(self.d1.kurtosis, 3.245089300687639)
        self.assertEqual(self.d1.median, 1.1774100225154747)
        self.assertEqual(self.d1.mode, 1)
        self.assertEqual(self.d1.entropy, 0.9420342421707937)
        self.assertEqual(self.d1.name, "MyWeibull")
    
    def test_methods(self):
        
        self.assertListEqual(list(self.d1.pdf([1,2])), [0.6065306597126334, 0.2706705664732254])
        self.assertListEqual(list(self.d1.cdf([1,2])), [0.3934693402873666, 0.8646647167633873])

        
    
# DISCRETE DISTRIBUTIONS
    
class Poisson(unittest.TestCase):
    """
    Tests the class Poisson.
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        
        # Test AssertionError
        with self.assertRaises(AssertionError):
            dis.Poisson(lmbda=-1., name="MyPoisson")
        
        # For the later tests
        self.d1 = dis.Poisson(lmbda=0.5, name="MyPoisson")
        
    def tearDown(self):
        pass
    
    def test_attributes(self):
        self.assertEqual(self.d1.type, 'Poisson')
        self.assertEqual(self.d1.support, 'N')
        self.assertEqual(self.d1.lmbda, 0.5)
        self.assertEqual(self.d1.mean, 0.5)
        self.assertEqual(self.d1.variance, 0.5)
        self.assertEqual(self.d1.std, 0.7071067811865476)
        self.assertEqual(self.d1.skewness, 1.414213562373095)
        self.assertEqual(self.d1.kurtosis, 4.414213562373095)
        self.assertEqual(self.d1.median, 0.)
        self.assertEqual(self.d1.k_max, 1000)
        self.assertEqual(self.d1.mode, 0.)
        self.assertEqual(self.d1.entropy, 0.8465735902799727)
        self.assertEqual(self.d1.name, "MyPoisson")
        
    def test_methods(self):
        
        self.assertListEqual(list(self.d1.pmf([1,2])), [0.3032653298563167, 0.07581633246407918])
        self.assertListEqual(list(self.d1.cdf([1,2])), [0.9097959895689501, 0.9856123220330293])

        
        
class Binomial(unittest.TestCase):
    """
    Tests the class Binomial.
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        
        # Test AssertionError
        with self.assertRaises(AssertionError):
            dis.Binomial(n=1., p=0.5)
        with self.assertRaises(AssertionError):
            dis.Binomial(n=-10, p=0.5)
        with self.assertRaises(AssertionError):
            dis.Binomial(n=10, p=1.5)
        with self.assertRaises(AssertionError):
            dis.Binomial(n=10, p=-0.5)
        
        # For the later tests
        self.d1 = dis.Binomial(n=10, p=0.7, name="MyBinomial")
        
    def tearDown(self):
        pass
    
    def test_attributes(self):
        
        self.assertEqual(self.d1.type, 'Binomial')
        self.assertEqual(self.d1.support, '{0,1,...,n}')
        
        self.assertEqual(self.d1.n, 10)
        self.assertEqual(self.d1.p, 0.7)
        self.assertAlmostEqual(self.d1.q, 0.3)
        
        self.assertEqual(self.d1.mean, 7.0)
        self.assertEqual(self.d1.variance, 2.1000000000000005)
        self.assertEqual(self.d1.std, 1.449137674618944)
        self.assertEqual(self.d1.skewness, -0.27602622373694163)
        self.assertEqual(self.d1.kurtosis, 2.876190476190476)
        self.assertEqual(self.d1.median, 7.0)
        
        self.assertEqual(self.d1.mode, 7.0)
        self.assertEqual(self.d1.entropy, 2.58229024912634)
        self.assertEqual(self.d1.name, "MyBinomial")
        
    def test_methods(self):
        
        self.assertListEqual(list(self.d1.pmf([1,2])), [0.00013778100000000018, 0.0014467005000000015])
        self.assertListEqual(list(self.d1.cdf([1,2])), [0.00014368590000000018, 0.0015903864000000017])
    

    
    
    
    
    
if __name__ == '__main__':
    unittest.main()
    
    
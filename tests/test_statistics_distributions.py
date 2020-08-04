# Solving relative path problem
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

# Import Unittest
import unittest

# Import my package
from scifin.statistics import distributions as dis
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#






class TestNormal(unittest.TestCase):
    """
    Tests the class Normal.
    """
    
    @classmethod
    def setUpClass(cls):
        #print('setupClass')
        pass
    
    
    @classmethod
    def tearDownClass(cls):
        #print('tearDownClass')
        pass
    
    
    def setUp(self):
        #print('setUp')
        
        # Testing sigma=0 case raises an AssertionError
        with self.assertRaises(AssertionError):
            dis.Normal(mu=1., sigma=0., name="MyGaussian")
        
        # For the later tests
        self.n1 = dis.Normal(mu=1., sigma=0.3, name="MyGaussian")
        
    
    def tearDown(self):
        # print('tearDown')
        pass
    

    def test_pdf(self):
        # print('test_pdf')
        self.assertEqual(self.n1.mu, 1.)
        self.assertEqual(self.n1.sigma, 0.3)
        self.assertEqual(self.n1.name, 'MyGaussian')
        self.assertEqual(self.n1.variance, 0.09)
        
        # self.n1.sigma = 0.2
        # self.assertEqual(self.n1.sigma, 0.2)
        # self.assertEqual(self.n1.variance, 0.04)
        
    
    
    
    
    

    
if __name__ == '__main__':
    unittest.main()
    
    
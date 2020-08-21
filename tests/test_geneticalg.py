# Solving relative path problem
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

# Import Unittest
import unittest

# Import third party packages
import pandas as pd

# Import my package
from scifin.geneticalg import geneticalg as gen
    

#---------#---------#---------#---------#---------#---------#---------#---------#---------#




class TestIndividual(unittest.TestCase):
    """
    Tests the class Individual.
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
        
        # Testing initialization
        self.i1 = gen.Individual(genes=[1,2,3], birth_date="2020-08-21", name="Albert")
        
        
    def tearDown(self):
        # print('tearDown')
        pass

        
    def test_init(self):
        self.assertEqual(self.i1.genes_names, None)
        self.assertListEqual(list(self.i1.genes), [1, 2, 3])
        self.assertEqual(self.i1.birth_date, "2020-08-21")
        self.assertEqual(self.i1.ngenes, 3)
        self.assertEqual(self.i1.name, "Albert")


    
class TestPopulation(unittest.TestCase):
    """
    Tests the class Population.
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
        
        # Testing initialization
        test_df = pd.DataFrame(columns=['g1','g2','g3'], index=['i1','i2'])
        test_df.loc['i1'] = {'g1':1, 'g2':2, 'g3':3}
        test_df.loc['i2'] = {'g1':4, 'g2':5, 'g3':6}
        self.p1 = gen.Population(df=test_df, n_genes=test_df.shape[1], name="MyPopulation")
        
        
    def tearDown(self):
        # print('tearDown')
        pass

        
    def test_init(self):
        self.assertListEqual(self.p1.data.index.tolist(), ['i1','i2'])
        self.assertListEqual(self.p1.data.iloc[0].tolist(), [1,2,3])
        self.assertListEqual(self.p1.data.iloc[1].tolist(), [4,5,6])
        self.assertEqual(self.p1.n_indiv, 2)
        self.assertEqual(self.p1.n_genes, 3)
        self.assertEqual(self.p1.name, "MyPopulation")   
        self.assertEqual(self.p1.history, None)
    

    
if __name__ == '__main__':
    unittest.main()
    

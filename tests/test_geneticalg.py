import pandas as pd

from scifin.geneticalg import geneticalg as gen


class TestIndividual:
    """
    Tests the class Individual.
    """

    def test_Individual_init(self):
        # Define Individual
        i1 = gen.Individual(genes=[1, 2, 3], birth_date="2020-08-21", name="Albert")

        # Test attributes values
        assert i1.genes_names is None
        assert list(i1.genes) == [1, 2, 3]
        assert i1.birth_date == "2020-08-21"
        assert i1.ngenes == 3
        assert i1.name == "Albert"


class TestPopulation:
    """
    Tests the class Population.
    """

    def test_Population_init(self):
        # Define Population
        test_df = pd.DataFrame(columns=['g1', 'g2', 'g3'], index=['i1', 'i2'])
        test_df.loc['i1'] = {'g1': 1, 'g2': 2, 'g3': 3}
        test_df.loc['i2'] = {'g1': 4, 'g2': 5, 'g3': 6}
        p1 = gen.Population(df=test_df, n_genes=test_df.shape[1], name="MyPopulation")

        # Test attributes values
        assert p1.data.index.tolist() == ['i1', 'i2']
        assert p1.data.iloc[0].tolist() == [1, 2, 3]
        assert p1.data.iloc[1].tolist() == [4, 5, 6]
        assert p1.n_indiv == 2
        assert p1.n_genes == 3
        assert p1.name == "MyPopulation"
        assert p1.history is None

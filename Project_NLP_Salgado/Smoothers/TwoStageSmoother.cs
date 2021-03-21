using System;

namespace Project_NLP_Salgado
{
    public class TwoStageSmoother : ISmoother
    {
        public INGramLanguageModel CollectionLevelLanguageModel { get; set; }
        public double L { get; set; }
        public double M { get; set; }

        public double ComputeSmoothedWordProbability(string u, string v, string w, int nGramCount, int n_1_gramCount, int uniqueNGrams)
        {
            var collectionProb = CollectionLevelLanguageModel.ComputeWordProbability(null, null, w);
            var dirichletProb = (nGramCount + M * collectionProb) / (n_1_gramCount + M);
            
            return (1 - L) * dirichletProb + L * collectionProb;
        }
    }
}
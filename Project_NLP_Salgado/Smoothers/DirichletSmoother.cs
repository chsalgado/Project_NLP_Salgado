using System;

namespace Project_NLP_Salgado
{
    public class DirichletSmoother : ISmoother
    {
        public INGramLanguageModel CollectionLevelLanguageModel { get; set; }
        public double M { get; set; }

        public double ComputeSmoothedWordProbability(string u, string v, string w, int nGramCount, int n_1_gramCount, int uniqueNGrams)
        {
            return (nGramCount + M * CollectionLevelLanguageModel.ComputeWordProbability(null, null, w)) / (n_1_gramCount + M);
        }
    }
}
using System;

namespace Project_NLP_Salgado
{
    public class AbsoluteDiscountSmoother : ISmoother
    {
        public INGramLanguageModel CollectionLevelLanguageModel { get; set; }
        public double D { get; set; }

        public double ComputeSmoothedWordProbability(string u, string v, string w, int nGramCount, int n_1_gramCount, int uniqueNGrams)
        {
            return (Math.Max(nGramCount - D, 0) + D * uniqueNGrams * CollectionLevelLanguageModel.ComputeWordProbability(null, null, w)) / n_1_gramCount;
        }
    }
}
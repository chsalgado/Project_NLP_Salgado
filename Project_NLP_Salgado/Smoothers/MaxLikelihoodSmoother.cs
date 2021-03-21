namespace Project_NLP_Salgado
{
    public class MaxLikelihoodSmoother : ISmoother
    {
        public double K { get; set; }

        public double ComputeSmoothedWordProbability(string u, string v, string w, int nGramCount, int n_1_gramCount)
        {
            return (nGramCount * 1.0) / (n_1_gramCount);
        }
    }
}
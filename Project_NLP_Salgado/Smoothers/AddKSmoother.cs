namespace Project_NLP_Salgado
{
    public class AddKSmoother : ISmoother
    {
        public double K { get; set; }

        public double ComputeSmoothedWordProbability(string u, string v, string w, int nGramCount, int n_1_gramCount)
        {
            return (nGramCount * 1.0 + K) / (n_1_gramCount + K * Corpus.ValidVocabulary.Count + 2); // Include STOP and UNK symbols
        }
    }
}
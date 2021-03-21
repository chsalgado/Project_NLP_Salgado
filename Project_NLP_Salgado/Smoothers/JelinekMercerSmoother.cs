namespace Project_NLP_Salgado
{
    public class JelinekMercerSmoother : ISmoother
    {
        public INGramLanguageModel CollectionLevelLanguageModel { get; set; }
        public double L { get; set; }

        public double ComputeSmoothedWordProbability(string u, string v, string w, int nGramCount, int n_1_gramCount)
        {
            double qMlWcat = (nGramCount * 1.0) / (n_1_gramCount);
            qMlWcat = double.IsFinite(qMlWcat) ? qMlWcat : 0.0;

            double qMlWcollection = CollectionLevelLanguageModel.ComputeWordProbability(null, null, w);
            qMlWcollection = double.IsFinite(qMlWcollection) ? qMlWcollection : 0.0;

            return (1 - L) * qMlWcat + L * qMlWcollection;
        }
    }
}
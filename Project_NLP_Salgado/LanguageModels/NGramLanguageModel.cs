using System.Collections.Generic;

namespace Project_NLP_Salgado
{
    public abstract class NGramLanguageModel : INGramLanguageModel
    {
        public ISmoother Smoother { get; set; }

        protected IDictionary<string, int> NGramCounts = new Dictionary<string, int>();

        public abstract void TrainLanguageModel(Corpus corpus);

        public abstract double CalculateDocumentLogProbability(Corpus corpus);

        public abstract double ComputeWordProbability(string u, string v, string w);
    }
}

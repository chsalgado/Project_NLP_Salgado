using System.Collections.Generic;

namespace Project_NLP_Salgado
{
    public abstract class NGramLanguageModel : INGramLanguageModel
    {
        protected IDictionary<string, int> NGramCounts = new Dictionary<string, int>();

        public abstract void TrainLanguageModel(Corpus corpus);

        public abstract double CalculateDocumentLogProbability(Corpus corpus, double kSmoothingValue, int validVocabularySize);

        public abstract double ComputeWordProbability(string u, string v, string w, double kSmoothingValue, int validVocabularySize);

        protected double ComputeWordProbabilityWithAddKSmooth(int nGramCount, int n_1_gramCount, double kSmoothingValue, int validVocabularySize)
        {
            return (nGramCount * 1.0 + kSmoothingValue) / (n_1_gramCount + kSmoothingValue * validVocabularySize);
        }
    }
}

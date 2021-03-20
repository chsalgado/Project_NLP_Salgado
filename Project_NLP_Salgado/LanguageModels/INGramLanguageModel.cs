using System;
using System.Collections.Generic;
using System.Text;

namespace Project_NLP_Salgado
{
    public interface INGramLanguageModel
    {
        void TrainLanguageModel(Corpus corpus);

        double CalculateDocumentLogProbability(Corpus corpus, double kSmoothingValue, int validVocabularySize);

        double ComputeWordProbability(string u, string v, string w, double kSmoothingValue, int validVocabularySize);
    }
}

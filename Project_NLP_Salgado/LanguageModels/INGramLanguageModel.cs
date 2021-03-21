using System;
using System.Collections.Generic;
using System.Text;

namespace Project_NLP_Salgado
{
    public interface INGramLanguageModel
    {
        void TrainLanguageModel(Corpus corpus);

        double CalculateDocumentLogProbability(Corpus corpus);

        double ComputeWordProbability(string u, string v, string w);
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace Project_NLP_Salgado
{
    public interface ISmoother
    {
        double ComputeSmoothedWordProbability(string u, string v, string w, int nGramCount, int n_1_gramCountvalidVocabularySize, int uniqueNGrams);
    }
}
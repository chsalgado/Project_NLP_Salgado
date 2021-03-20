using System;
using System.Linq;
using System.Text.RegularExpressions;

namespace Project_NLP_Salgado
{
    public class UnigramLanguageModel : NGramLanguageModel
    {
        public override void TrainLanguageModel(Corpus trainingCorpus)
        {
            // We are fine flattening all setences into one big string as every word probability is independent of any previous one, so no risk on wrapping sentences
            // Enumerate once so we don't keep on doing it later on
            var flattenedTokenizedAndProcessedSentences = trainingCorpus.AllTokenizedSentences.SelectMany(s => s).ToList();

            // Add n-gram counts (used in numerator)
            this.NGramCounts = flattenedTokenizedAndProcessedSentences.GroupBy(w => w).ToDictionary(g => new Unigram { w = g.Key }.GetComparisonKey(), g => g.Count());

            // Add n-1-gram counts (used in denominator)
            // In the case of unigrams, we care about total token counts, including STOP tokens
            this.NGramCounts[new Unigram { w = string.Empty }.GetComparisonKey()] = trainingCorpus.TotalWordsCount;
        }

        public override double CalculateDocumentLogProbability(Corpus testCorpus, double kSmoothingValue = 0.0, int validVocabularySize = 0)
        {
            double corpusLogP = 0.0;

            // Sentence probability = multiplication of all words probabilities
            foreach (var sentence in testCorpus.AllTokenizedSentences)
            {
                double logPs = 0.0;

                foreach (var w in sentence)
                {
                    double qW = ComputeWordProbability(string.Empty, string.Empty, w, kSmoothingValue, validVocabularySize);

                    // Add to sentence probability
                    logPs += Math.Log2(qW);
                }

                corpusLogP += logPs;
            }

            return corpusLogP;
        }

        public override double ComputeWordProbability(string u, string v, string w, double kSmoothingValue, int validVocabularySize)
        {
            // Total token count in training to compute q(w)
            int totalTokenCount = this.NGramCounts[new Unigram { w = string.Empty }.GetComparisonKey()];

            this.NGramCounts.TryGetValue(new Unigram { w = w }.GetComparisonKey(), out int wCount);

            // q(w) = c(w)/c()
            // q(w)_{addK} = (c(w) + k)/(c() + k|V*|) 
            double qW = ComputeWordProbabilityWithAddKSmooth(wCount, totalTokenCount, kSmoothingValue, validVocabularySize);

            return qW;
        }
    }
}

using System;
using System.Linq;

namespace Project_NLP_Salgado
{
    public class BigramLanguageModel : NGramLanguageModel
    {
        public override void TrainLanguageModel(Corpus trainingCorpus)
        {
            // We need to process sentece by sentence to avoid wrapping sentences, ie. counting (STOP, w) bigrams
            foreach (var sentence in trainingCorpus.AllTokenizedSentences)
            {
                // Initialize x_{-1} to START
                var v = "<s>";

                // We now need to store all counts of c(v, w) and c(v)
                foreach (var w in sentence)
                {
                    Unigram vUnigram = new Unigram { w = v };
                    Bigram vwBigram = new Bigram { v = v, w = w };

                    // +1 to current count, current will be 0 if not found, thus starting at 1 as expected
                    this.NGramCounts.TryGetValue(vUnigram.GetComparisonKey(), out int vCount);
                    vCount++;
                    this.NGramCounts[vUnigram.GetComparisonKey()] = vCount;

                    this.NGramCounts.TryGetValue(vwBigram.GetComparisonKey(), out int vwCount);
                    vwCount++;
                    this.NGramCounts[vwBigram.GetComparisonKey()] = vwCount;

                    // Replace previous token
                    v = w;
                }
            }
        }

        public override double CalculateDocumentLogProbability(Corpus testCorpus)
        {
            double corpusLogP = 0.0;

            // Sentence probability = multiplication of all words probabilities
            foreach (var sentence in testCorpus.AllTokenizedSentences)
            {
                // Initialize x_{-1} to START
                var v = "<s>";

                double logPs = 0.0;

                foreach (var w in sentence)
                {
                    double qWv = ComputeWordProbability(string.Empty, v, w);

                    // Add to sentence probability
                    logPs += Math.Log2(qWv);

                    // Replace previous token
                    v = w;
                }

                corpusLogP += logPs;
            }

            return corpusLogP;
        }

        public override double ComputeWordProbability(string u, string v, string w)
        {
            Unigram vUnigram = new Unigram { w = v };
            Bigram vwBigram = new Bigram { v = v, w = w };

            // Compute word probability given the previous one
            this.NGramCounts.TryGetValue(vUnigram.GetComparisonKey(), out int vCount);
            this.NGramCounts.TryGetValue(vwBigram.GetComparisonKey(), out int vwCount);

            // q(w|v) = c(v, w)/c(v)
            // q(w|v)_{addK} = (c(v, w) + k)/(c(v) + k|V*|) 
            double qWv = this.Smoother.ComputeSmoothedWordProbability(u,v, w, vwCount, vCount);

            return qWv;
        }
    }
}

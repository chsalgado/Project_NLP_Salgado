using System;
using System.Linq;

namespace Project_NLP_Salgado
{
    public class TrigramLanguageModel : NGramLanguageModel
    {
        public override void TrainLanguageModel(Corpus trainingCorpus)
        {
            // We need to process sentece by sentence to avoid wrapping sentences, ie. counting (STOP, v, w) trigrams
            foreach (var sentence in trainingCorpus.AllTokenizedSentences)
            {
                // Initialize x_{-1}, x_{-2} to START
                var u = "<s>";
                var v = "<s>";

                // We now need to store all counts of c(u, v, w) and c(v, w)
                foreach (var w in sentence)
                {
                    Bigram uvBigram = new Bigram {  v = u, w = v };
                    Trigram uvwTrigram = new Trigram { u = u, v = v, w = w };

                    // +1 to current count, current will be 0 if not found, thus starting at 1 as expected
                    this.NGramCounts.TryGetValue(uvBigram.GetComparisonKey(), out int uvCount);
                    uvCount++;
                    this.NGramCounts[uvBigram.GetComparisonKey()] = uvCount;

                    var isNewNgram = !this.NGramCounts.TryGetValue(uvwTrigram.GetComparisonKey(), out int uvwCount);
                    uvwCount++;
                    this.NGramCounts[uvwTrigram.GetComparisonKey()] = uvwCount;

                    if (isNewNgram)
                    {
                        this.UniqueNGramsCount++;
                    }

                    // Replace previous tokens
                    u = v;
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
                // Initialize x_{-1}, x_{-2} to START
                var u = "<s>";
                var v = "<s>";
                
                double logPs = 0.0;

                foreach (var w in sentence)
                {
                    double qWuv = ComputeWordProbability(u, v, w);

                    // Add to sentence probability
                    logPs += Math.Log2(qWuv);

                    // Replace previous tokens
                    u = v;
                    v = w;
                }

                corpusLogP += logPs;
            }

            return corpusLogP;
        }

        public override double ComputeWordProbability(string u, string v, string w)
        {
            Bigram uvBigram = new Bigram { v = u, w = v };
            Trigram uvwTrigram = new Trigram { u = u, v = v, w = w }; ;

            // Compute word probability given the previous two
            this.NGramCounts.TryGetValue(uvBigram.GetComparisonKey(), out int uvCount);
            this.NGramCounts.TryGetValue(uvwTrigram.GetComparisonKey(), out int uvwCount);

            // q(w|u, v) = c(u, v, w)/c(u, v)
            // q(w|u, v)_{addK} = (c(u, v, w) + k)/(c(u, v) + k|V*|) 
            double qWuv = this.Smoother.ComputeSmoothedWordProbability(u, v, w, uvwCount, uvCount, this.UniqueNGramsCount);

            return qWuv;
        }
    }
}

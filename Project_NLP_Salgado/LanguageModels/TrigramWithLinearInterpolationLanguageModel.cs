using System;
using System.Linq;

namespace Project_NLP_Salgado
{
    public class TrigramWithLinearInterpolationLanguageModel : NGramLanguageModel
    {
        private UnigramLanguageModel UnigramLM;
        private BigramLanguageModel BigramLM;
        private TrigramLanguageModel TrigramLM;
        private double L1;
        private double L2;
        private double L3;

        public TrigramWithLinearInterpolationLanguageModel(double l1, double l2, double l3, ISmoother smoother)
        {
            UnigramLM = new UnigramLanguageModel { Smoother = smoother };
            BigramLM = new BigramLanguageModel { Smoother = smoother };
            TrigramLM = new TrigramLanguageModel { Smoother = smoother };
            Smoother = smoother;
            L1 = l1;
            L2 = l2;
            L3 = l3;
        }

        public override void TrainLanguageModel(Corpus trainingCorpus)
        {
            // Since we interpolate all three q's for every n-gram, preparing this model means preparing all the base n-gram models
            this.UnigramLM.TrainLanguageModel(trainingCorpus);
            this.BigramLM.TrainLanguageModel(trainingCorpus);
            this.TrigramLM.TrainLanguageModel(trainingCorpus);
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
            double qW =  this.UnigramLM.ComputeWordProbability(string.Empty, string.Empty, w);
            qW = double.IsFinite(qW) ? qW : 0.0;

            double qWv = this.BigramLM.ComputeWordProbability(string.Empty, v, w);
            qWv = double.IsFinite(qWv) ? qWv : 0.0;

            double qWuv = this.TrigramLM.ComputeWordProbability(u, v, w);
            qWuv = double.IsFinite(qWuv) ? qWuv : 0.0;

            return this.L1 * qW + this.L2 * qWv + this.L3 * qWuv;
        }
    }
}

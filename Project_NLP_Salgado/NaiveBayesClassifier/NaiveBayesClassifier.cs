using System;
using System.Collections.Generic;
using System.Linq;

namespace Project_NLP_Salgado
{
    public static class NaiveBayesClassifier
    {
        private static int TotalTrainingDocuments { get; set; }
        private static IDictionary<string, int> CategoryTrainingDocumentsCount { get; set; }

        public static void InitializeAndFillCategoryTrainingCounts(IDictionary<string, HashSet<string>> categoriesMap)
        {
            CategoryTrainingDocumentsCount = new Dictionary<string, int>();
            HashSet<string> allTrainingDocs = new HashSet<string>();
            foreach (var kvp in categoriesMap)
            {
                var trainingDocumentsForCategory = kvp.Value.Where(d => d.StartsWith("training"));
                CategoryTrainingDocumentsCount[kvp.Key] = trainingDocumentsForCategory.Count();

                allTrainingDocs.UnionWith(trainingDocumentsForCategory);
            }

            TotalTrainingDocuments = allTrainingDocs.Count;
        }
        
        public static string ClassifyDocument(string documentPath, LanguageModelHyperparameters hyperparameters)
        {
            var testCorpus = new Corpus();
            testCorpus.InitializeAndPreprocessDocument(documentPath, hyperparameters.IgnoreCase);

            TextProcessingUtilities.UnkCorpus(testCorpus, Corpus.ValidVocabulary);
            TextProcessingUtilities.AddStopTokens(testCorpus);

            (string category, double logProbability) categoryWithHighestLogProbability = (string.Empty, double.NegativeInfinity);

            foreach (var categoryLanguageModel in hyperparameters.CategoryNGramLanguageModelsMap)
            {
                // P(c_i) = N_i / N
                var categoryLogProbability = Math.Log2(CategoryTrainingDocumentsCount[categoryLanguageModel.Key] * 1.0 / TotalTrainingDocuments);
                // mult(P(d_j | c_i)) = P(d | c_i)
                var documentLogProbabilityForCategory = categoryLanguageModel.Value.CalculateDocumentLogProbability(testCorpus);

                // P(c_i | d) = mult(P(d_j | c_i)) P(c_i)
                // P(c_i | d) = P(d | c_i) P(c_i)
                var categoryLogProbabilityForDocument = documentLogProbabilityForCategory + categoryLogProbability;

                // category = argmax_i mult(P(d_j | c_i)) P(c_i)
                if (categoryLogProbabilityForDocument > categoryWithHighestLogProbability.logProbability)
                {
                    categoryWithHighestLogProbability = (categoryLanguageModel.Key, categoryLogProbabilityForDocument);
                }
            }

            return categoryWithHighestLogProbability.category;
        }
    }
}

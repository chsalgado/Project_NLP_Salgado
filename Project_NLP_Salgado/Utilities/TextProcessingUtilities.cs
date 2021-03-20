using System;
using System.Collections.Generic;
using System.Linq;

namespace Project_NLP_Salgado
{
    public static class TextProcessingUtilities
    {
        public static void AddStopTokens(Corpus corpus)
        {
            foreach (var line in corpus.AllTokenizedSentences)
            {
                line.Add("</s>");
            };

            corpus.ComputeTotalWordsCount();
        }

        public static void UnkCorpus(Corpus corpus, HashSet<string> validVocabulary)
        {
            foreach (var line in corpus.AllTokenizedSentences)
            {
                for (var i = 0; i < line.Count; i++)
                {
                    if (!validVocabulary.Contains(line[i])) 
                    {
                        line[i] = "<unk>";
                    }
                }
            };
        }

        public static HashSet<string> GetValidVocabulary(Corpus corpus, double unkRatio)
        {
            // Implement a naive unk strategy: unk top n% of lowest count words
            // TODO parametize the strategy

            // Order by word count, then alphabetically to ensure determinism TODO I may want to randomize
            var uniqueSortedTokens = corpus.AllTokenizedSentences.SelectMany(s => s).GroupBy(w => w).OrderByDescending(g => g.Count()).ThenBy(g => g.Key);

            // Remove n% of words
            var ratioToKeep = 1.0 - unkRatio;
            var wordsToKeep = (int)Math.Floor(uniqueSortedTokens.Count() * ratioToKeep);

            var validVocabulary = uniqueSortedTokens.Take(wordsToKeep).Select(g => g.Key).ToHashSet();

            return validVocabulary;
        }
    }
}

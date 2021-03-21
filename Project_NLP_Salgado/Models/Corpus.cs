using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Project_NLP_Salgado
{
    public class Corpus
    {
        public static IDictionary<string, HashSet<string>> CategoriesMap { get; set; }
        public static HashSet<string> ValidVocabulary { get; set; }

        public static void InitializeAndFillCategoriesMap(string path)
        {
            var fileToCategoriesRawLines = File.ReadAllLines(Path.Combine(path, "Dataset/reuters/cats.txt"));

            CategoriesMap = new Dictionary<string, HashSet<string>>();
            foreach (var fileToCategoryLine in fileToCategoriesRawLines)
            {
                var splittedLine = fileToCategoryLine.Split();
                var fileId = splittedLine[0];
                var catergories = splittedLine[1..];

                foreach(var category in catergories)
                {
                    if(!CategoriesMap.ContainsKey(category))
                    {
                        CategoriesMap[category] = new HashSet<string>();
                    }

                    CategoriesMap[category].Add(fileId);
                }
            }
        }

        public static void InitializeAndFillValidVocabulary(Corpus corpus, LanguageModelHyperparameters hyperparameters)
        {
            // Get our valid vocabulary by removing the bottom x% of words by frequency
            Corpus.ValidVocabulary = TextProcessingUtilities.GetValidVocabulary(corpus, hyperparameters.UnkRatio);
        }

        public IList<List<string>> AllTokenizedSentences { get; set; }

        public string Category { get; set; }

        public int TotalWordsCount { get; set; }

        public void InitializeAndPreprocessCategoryCorpus(string path, string category, LanguageModelHyperparameters hyperparameters)
        {
            Category = category;

            var allDocumentPaths = Directory.GetFiles(path);

            AllTokenizedSentences = new List<List<string>>();
            foreach (var documentPath in allDocumentPaths)
            {
                var documentName = $@"{new DirectoryInfo(documentPath).Parent.Name}/{Path.GetFileName(documentPath)}";

                if (category.Equals("ALLCATEGORIES") || CategoriesMap[category].Contains(documentName))
                {
                    var document = new Document();
                    document.InitializeAndPreprocess(documentPath, hyperparameters.IgnoreCase);

                    AllTokenizedSentences = AllTokenizedSentences.Concat(document.AllTokenizedSentences).ToList();
                }
            }

            ComputeTotalWordsCount();
        }

        public void InitializeAndPreprocessDocument(string documentPath, bool ignoreCasing)
        {
            AllTokenizedSentences = new List<List<string>>();

            var document = new Document();
            document.InitializeAndPreprocess(documentPath, ignoreCasing);

            AllTokenizedSentences = AllTokenizedSentences.Concat(document.AllTokenizedSentences).ToList();

            ComputeTotalWordsCount();
        }

        public void ComputeTotalWordsCount()
        {
            // Do not count start tags as part of total count, as the start tag is not part of the valid vocabulary
            // Start tag should not have a probability in any N-gram, thus it should not be included in word count, else average log probability will use in in denominator, but not in numerator
            TotalWordsCount = AllTokenizedSentences.SelectMany(s => s).Where(w => !w.Equals("<s>")).Count();
        }
    }

    public class Document
    {
        static char[] delimiters = new char[] { ' ', '\r', '\n' };
        
        public IList<List<string>> AllTokenizedSentences { get; set; }

        public void InitializeAndPreprocess(string path, bool ignoreCasing)
        {
            // Any preprocessing not having to do with stop/end tags (LM owns that logic) and unking goes here

            // TODO maybe lowercase;
            // Enumerate as soon as we are done parsing to avoid deferred execution issues
            var allDocumentLines = File.ReadAllLines(path);
            
            AllTokenizedSentences = allDocumentLines.Select(l =>
            {
                l = ignoreCasing ? l.ToLower() : l;
                var tokenizedSentence = l.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);

                return tokenizedSentence.ToList();
            }).ToList();
        }
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;

namespace Project_NLP_Salgado
{
    class Program
    {
        static void Main(string[] args)
        {
            var appConfigName = "app.config";
            try
            {
                if (!File.Exists(Path.Combine(args[0], appConfigName)))
                {
                    Console.WriteLine($"{appConfigName} not found in {args[0]}");
                }
            }
            catch (Exception)
            {
                Console.WriteLine($"{appConfigName} not found in {args[0]}");
            }

            var allRuns = File.ReadAllLines(Path.Combine(args[0], appConfigName)).Where(s => !string.IsNullOrWhiteSpace(s) && !s.StartsWith("##"));
            var allHyperparameters = allRuns.Select(r => LanguageModelHyperparameters.GenerateFromArguments(r));

            // Our corpus existing classification is independent of training
            Corpus.InitializeAndFillCategoriesMap(args[0]);
            NaiveBayesClassifier.InitializeAndFillCategoryTrainingCounts(Corpus.CategoriesMap);

            foreach (var hyperparameters in allHyperparameters)
            {
                // We do this here as volcabulary can change depending on hyperparams
                Corpus.InitializeAndFillValidVocabulary(Path.Combine(args[0], "Dataset/reuters/training"), hyperparameters);
                TrainAllLanguageModels(hyperparameters, args[0]);

                ClassifyAllTestDocuments(hyperparameters, args[0]);
            }
        }

        private static void TrainAllLanguageModels(LanguageModelHyperparameters hyperparameters, string basePath)
        {
            var stopwatch = new Stopwatch();
            var globalStopwatch = new Stopwatch();
            
            var i = 1;
            globalStopwatch.Start();
            foreach (var categoryLanguageModel in hyperparameters.CategoryNGramLanguageModelsMap)
            {
                stopwatch.Restart();
                var categoryTrainingCorpus = new Corpus();
                categoryTrainingCorpus.InitializeAndPreprocessCategoryCorpus(Path.Combine(basePath, "Dataset/reuters/training"), categoryLanguageModel.Key, hyperparameters);

                TextProcessingUtilities.UnkCorpus(categoryTrainingCorpus, Corpus.ValidVocabulary);
                TextProcessingUtilities.AddStopTokens(categoryTrainingCorpus);

                categoryLanguageModel.Value.TrainLanguageModel(categoryTrainingCorpus);

                stopwatch.Stop();
                Console.WriteLine($@"LanguageModel for category {categoryLanguageModel.Key} trained in {stopwatch.ElapsedMilliseconds} ms. {i}/{hyperparameters.CategoryNGramLanguageModelsMap.Count} done");

                i++;
            }

            globalStopwatch.Stop();
            Console.WriteLine();
            Console.WriteLine($@"Training done in {globalStopwatch.ElapsedMilliseconds} ms");
        }

        private static void ClassifyAllTestDocuments(LanguageModelHyperparameters hyperparameters, string basePath)
        {
            var pathForTestingDocuments = Path.Combine(basePath, "Dataset/reuters/", hyperparameters.TestCorpusTag);
            var documentPaths = Directory.GetFiles(pathForTestingDocuments);
            var correctlyClassifiedDocuments = 0;

            var allPredictions = new List<string>();
            var processedDocuments = 0;
            foreach (var documentPath in documentPaths)
            {
                var predictedCategory = NaiveBayesClassifier.ClassifyDocument(documentPath, hyperparameters);
                processedDocuments++;

                if (processedDocuments % 100 == 0)
                {
                    Console.WriteLine($@"Processed {processedDocuments} / {documentPaths.Length} documents");
                }

                var documentName = $@"{new DirectoryInfo(documentPath).Parent.Name}/{Path.GetFileName(documentPath)}";

                allPredictions.Add($@"{documentName} {predictedCategory}");
                if (Corpus.CategoriesMap[predictedCategory].Contains(documentName))
                {
                    correctlyClassifiedDocuments++;
                }
            }

            File.WriteAllLines(Path.Combine(basePath, "predictions"), allPredictions);
            Console.WriteLine();
            Console.WriteLine($@"Correctly classified {correctlyClassifiedDocuments} / {documentPaths.Length} documents");
        }
    }

    public class LanguageModelHyperparameters
    {
        public IDictionary<string, INGramLanguageModel> CategoryNGramLanguageModelsMap { get; set; }
        public bool UseFullSet { get; set; }
        public string TestCorpusTag { get; set; }
        public double UnkRatio { get; set; }
        public double K { get; set; }
        public bool IgnoreCase { get; set; }
        public double L1 { get; set; }
        public double L2 { get; set; }
        public double L3 { get; set; }

        public static LanguageModelHyperparameters GenerateFromArguments(string args)
        {
            args = args.ToLower();
            var splittedArgs = args.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            
            // LM
            var nGramLanguageModels = Corpus.CategoriesMap.ToDictionary(cdp => cdp.Key, cdp => 
            {
                INGramLanguageModel modelToUse = null;
                switch (splittedArgs[Array.IndexOf(splittedArgs, "-lm") + 1])
                {
                    case "unigram":
                        modelToUse = new UnigramLanguageModel();
                        break;
                    case "bigram":
                        modelToUse = new BigramLanguageModel();
                        break;
                    case "trigram":
                        modelToUse = new TrigramLanguageModel();
                        break;
                    case "trigramwithlinearinterpolation":
                        modelToUse = new TrigramWithLinearInterpolationLanguageModel(
                            Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l1") + 1]),
                            Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l2") + 1]),
                            Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l3") + 1]));
                        break;
                }

                return modelToUse;
            });

            return new LanguageModelHyperparameters
            {
                CategoryNGramLanguageModelsMap = nGramLanguageModels,
                UseFullSet = Array.IndexOf(splittedArgs, "-usefullset") >= 0,
                TestCorpusTag = splittedArgs[Array.IndexOf(splittedArgs, "-testcorpustag") + 1],
                UnkRatio = Array.IndexOf(splittedArgs, "-unkratio") >= 0 ? Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-unkratio") + 1]) : 0.1,
                K = Array.IndexOf(splittedArgs, "-k") >= 0 ? Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-k") + 1]) : 0.0,
                IgnoreCase = Array.IndexOf(splittedArgs, "-ignorecase") >= 0,
                L1 = Array.IndexOf(splittedArgs, "-l1") >= 0 ? Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l1") + 1]) : 0.0,
                L2 = Array.IndexOf(splittedArgs, "-l2") >= 0 ? Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l2") + 1]) : 0.0,
                L3 = Array.IndexOf(splittedArgs, "-l3") >= 0 ? Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l3") + 1]) : 0.0,
            };
        }

        public override string ToString()
        {
            Type t = this.GetType();
            PropertyInfo[] pi = t.GetProperties();

            string stringToReturn = string.Empty;
            foreach (PropertyInfo p in pi)
            {
                stringToReturn += $"-{p.Name}: {p.GetValue(this)} ";
            }

            return stringToReturn;
        }
    }
}

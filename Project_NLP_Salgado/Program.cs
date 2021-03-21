using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Dynamic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
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
                var globalStopwatch = new Stopwatch();
                globalStopwatch.Start();

                // We do this here as volcabulary can change depending on hyperparams
                Console.WriteLine($@"Parsing all training documents to get valid vocabulary and train collection level unigram model (used by some smoothing techniques)...");
                var allCategoriesTrainingCorpus = new Corpus();
                allCategoriesTrainingCorpus.InitializeAndPreprocessCategoryCorpus(Path.Combine(args[0], "Dataset/reuters/training"), "ALLCATEGORIES", hyperparameters);
                Corpus.InitializeAndFillValidVocabulary(allCategoriesTrainingCorpus, hyperparameters);
                Console.WriteLine($@"Generated valid vocabulary. Elapsed time: {globalStopwatch.ElapsedMilliseconds}");

                TrainAllLanguageModels(hyperparameters, args[0], allCategoriesTrainingCorpus);
                
                Console.WriteLine();
                Console.WriteLine($@"Training done in {globalStopwatch.ElapsedMilliseconds} ms");

                Console.WriteLine();
                Console.WriteLine($@"Classifying documents"); 
                ClassifyAllTestDocuments(hyperparameters, args[0]);
                Console.WriteLine($@"Elapsed time: {globalStopwatch.ElapsedMilliseconds} ms");
            }
        }

        private static void TrainAllLanguageModels(LanguageModelHyperparameters hyperparameters, string basePath, Corpus preProcessedCollectionCorpus)
        {
            var stopwatch = new Stopwatch();
            
            var i = 1;
            foreach (var categoryLanguageModel in hyperparameters.CategoryNGramLanguageModelsMap.Append(new KeyValuePair<string, INGramLanguageModel>("ALLCATEGORIES", hyperparameters.CollectionLevelLanguageModel)))
            {
                var category = categoryLanguageModel.Key;
                var languageModel = categoryLanguageModel.Value;

                stopwatch.Restart();

                Corpus preProcessedCategoryTrainingCorpus;
                if (category.Equals("ALLCATEGORIES"))
                {
                    preProcessedCategoryTrainingCorpus = preProcessedCollectionCorpus;
                }
                else
                {
                    preProcessedCategoryTrainingCorpus = new Corpus();
                    preProcessedCategoryTrainingCorpus.InitializeAndPreprocessCategoryCorpus(Path.Combine(basePath, "Dataset/reuters/training"), category, hyperparameters);
                }

                TextProcessingUtilities.UnkCorpus(preProcessedCategoryTrainingCorpus, Corpus.ValidVocabulary);
                TextProcessingUtilities.AddStopTokens(preProcessedCategoryTrainingCorpus);

                languageModel.TrainLanguageModel(preProcessedCategoryTrainingCorpus);

                stopwatch.Stop();
                //Console.WriteLine($@"LanguageModel for category {category} trained in {stopwatch.ElapsedMilliseconds} ms. {i}/{hyperparameters.CategoryNGramLanguageModelsMap.Count} done");

                i++;
            }
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
                    // Console.WriteLine($@"Processed {processedDocuments} / {documentPaths.Length} documents");
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
            Console.WriteLine($@"Correctly classified {correctlyClassifiedDocuments} / {documentPaths.Length} documents ({correctlyClassifiedDocuments * 1.0 / documentPaths.Length})");
        }
    }

    public class LanguageModelHyperparameters
    {
        public IDictionary<string, INGramLanguageModel> CategoryNGramLanguageModelsMap { get; set; }
        public string TestCorpusTag { get; set; }
        public double UnkRatio { get; set; }
        public bool IgnoreCase { get; set; }
        public double L1 { get; set; }
        public double L2 { get; set; }
        public double L3 { get; set; }

        // Collection-level model used in some smoothing techniques
        public INGramLanguageModel CollectionLevelLanguageModel { get; set; }

        public static LanguageModelHyperparameters GenerateFromArguments(string args)
        {
            args = args.ToLower();
            var splittedArgs = args.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            // Smoothers
            // Create the collection-level unigram model with no smoothing (max-likelihood) used in some smoothing techniques
            INGramLanguageModel collectionLevelLanguageModel = new UnigramLanguageModel { Smoother = new MaxLikelihoodSmoother() };
            ISmoother smoother = null;
            switch (splittedArgs[Array.IndexOf(splittedArgs, "-smoothingtechnique") + 1])
            {
                case "ml":
                    smoother = new MaxLikelihoodSmoother();
                    break;
                case "addk":
                    smoother = new AddKSmoother { K = Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-k") + 1]) };
                    break;
                case "jm":
                    smoother = new JelinekMercerSmoother { CollectionLevelLanguageModel = collectionLevelLanguageModel, L = Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l") + 1]) };
                    break;
            }

            // LM
            var nGramLanguageModels = Corpus.CategoriesMap.ToDictionary(cdp => cdp.Key, cdp => 
            {
                INGramLanguageModel modelToUse = null;
                switch (splittedArgs[Array.IndexOf(splittedArgs, "-lm") + 1])
                {
                    case "unigram":
                        modelToUse = new UnigramLanguageModel { Smoother = smoother };
                        break;
                    case "bigram":
                        modelToUse = new BigramLanguageModel { Smoother = smoother };
                        break;
                    case "trigram":
                        modelToUse = new TrigramLanguageModel { Smoother = smoother };
                        break;
                    case "trigramwithlinearinterpolation":
                        modelToUse = new TrigramWithLinearInterpolationLanguageModel(
                            Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l1") + 1]),
                            Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l2") + 1]),
                            Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l3") + 1]),
                            smoother);
                        break;
                }

                return modelToUse;
            });

            return new LanguageModelHyperparameters
            {
                CategoryNGramLanguageModelsMap = nGramLanguageModels,
                TestCorpusTag = splittedArgs[Array.IndexOf(splittedArgs, "-testcorpustag") + 1],
                UnkRatio = Array.IndexOf(splittedArgs, "-unkratio") >= 0 ? Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-unkratio") + 1]) : 0.1,
                IgnoreCase = Array.IndexOf(splittedArgs, "-ignorecase") >= 0,
                L1 = Array.IndexOf(splittedArgs, "-l1") >= 0 ? Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l1") + 1]) : 0.0,
                L2 = Array.IndexOf(splittedArgs, "-l2") >= 0 ? Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l2") + 1]) : 0.0,
                L3 = Array.IndexOf(splittedArgs, "-l3") >= 0 ? Double.Parse(splittedArgs[Array.IndexOf(splittedArgs, "-l3") + 1]) : 0.0,
                CollectionLevelLanguageModel = collectionLevelLanguageModel,
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

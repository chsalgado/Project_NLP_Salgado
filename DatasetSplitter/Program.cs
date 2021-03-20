using System;
using System.IO;
using System.Linq;
using System.Xml.Linq;

namespace DatasetSplitter
{
    class Program
    {
        static void Main(string[] args)
        {
            var reutersDatasetPath = $@"C:\Users\chsalgad\source\repos\Project_NLP_Salgado\Project_NLP_Salgado\Config\Dataset";

            var allTopics = File.ReadAllLines(Path.Combine(reutersDatasetPath, "all-topics-strings.lc.txt")).ToHashSet();

            var sampleFile = File.ReadAllText(Path.Combine(reutersDatasetPath, "sgmFiles", "reut2-000.xml"));

            var xml = XDocument.Parse(sampleFile);

            Console.ReadLine();
        }
    }
}

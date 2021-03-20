namespace Project_NLP_Salgado
{
    public class Unigram : INGram
    {
        public string w { get; set; }

        public string GetComparisonKey()
        {
            return $"|{w}|";
        }
    }
}

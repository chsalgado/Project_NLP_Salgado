namespace Project_NLP_Salgado
{
    public class Bigram : INGram
    {
        public string v { get; set; }
        public string w { get; set; }

        public string GetComparisonKey()
        {
            return $"|{v}|{w}|";
        }
    }
}

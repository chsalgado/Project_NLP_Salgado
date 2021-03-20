namespace Project_NLP_Salgado
{
    public class Trigram : INGram
    {
        public string u { get; set; }
        public string v { get; set; }
        public string w { get; set; }

        public string GetComparisonKey()
        {
            return $"|{u}|{v}|{w}|";
        }
    }
}

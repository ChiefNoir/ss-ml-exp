using Microsoft.ML;
using Microsoft.ML.Data;
using System.Diagnostics;
using System.Text.Json;
using Bert;
using LLama;

namespace Mlexp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //var vocab = @"C:\Develop\ss-ml-exp\Mlexp\Assets\Vocabulary\bert-large-uncased-vocab.txt";
            //var modelPath = @"C:\Develop\ss-ml-exp\Mlexp\Assets\Models\bertsquad-10.onnx";

            //var model = new Bert.Bert(vocab, modelPath);

            //var (tokens, probability) = model.Predict
            //    (
            //        @"Jim woke up at 8 a.m.,drunk some black coffee and ate a sandwich with nails.
            //          He took his axe and went to the forest. In the woods he saw a hare, a chicken and Marry.
            //          He killed them all.",
            //        "Who killed Marry?"
            //    );

            //Console.WriteLine(JsonSerializer.Serialize(new
            //{
            //    Probability = probability,
            //    Tokens = tokens
            //}));

            //var cl = new LLamaMachine();
            //cl.DoChat();

            Sample ss = new Sample();
            ss.DoWork();
        }
    }
}
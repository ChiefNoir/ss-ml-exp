using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLama
{
    public class LLamaMachine
    {
        public void DoChat()
        {
            var modelPath = @"C:\Develop\_language-models\wizardLM-13B-Uncensored.ggml.q4_1.bin";

            var model = new LLamaModel(new LLamaParams(model: modelPath, n_ctx: 512, repeat_penalty: 1.0f));
            var session = new ChatSession<LLamaModel>(model)
                        //.WithPromptFile("<Your prompt file path>")
                            .WithAntiprompt(new string[] { "User:" });
            Console.Write("\nUser:");
            while (true)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                var question = Console.ReadLine();
                Console.ForegroundColor = ConsoleColor.White;
                var outputs = session.Chat(question); // It's simple to use the chat API.
                foreach (var output in outputs)
                {
                    Console.Write(output);
                }
            }
        }
    }
}

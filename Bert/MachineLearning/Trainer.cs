using Bert.MachineLearning.DataModel;
using Microsoft.ML;

namespace Bert.MachineLearning
{
    internal class Trainer
    {
        private readonly MLContext _mlContext;

        public Trainer()
        {
            _mlContext = new MLContext(11);
        }

        public ITransformer BuildAndTrain(string bertModelPath, bool useGPU)
        {
            var pipeline = _mlContext.Transforms
                            .ApplyOnnxModel(modelFile: bertModelPath,
                                            outputColumnNames: new[] { "unstack:1",
                                                                       "unstack:0",
                                                                       "unique_ids:0" },
                                            inputColumnNames: new[] {"unique_ids_raw_output___9:0",
                                                                      "segment_ids:0",
                                                                      "input_mask:0",
                                                                      "input_ids:0" },
                                            gpuDeviceId: useGPU ? 0 : (int?)null);

            return pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<BertInput>()));
        }
    }
}

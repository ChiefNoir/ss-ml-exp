using Bert.MachineLearning.DataModel;
using Microsoft.ML;

namespace Bert.MachineLearning
{
    internal class Predictor
    {
        private readonly MLContext _mLContext;
        private readonly PredictionEngine<BertInput, BertPredictions> _predictionEngine;

        public Predictor(ITransformer trainedModel)
        {
            _mLContext = new MLContext();
            _predictionEngine = _mLContext.Model
                                          .CreatePredictionEngine<BertInput, BertPredictions>(trainedModel);
        }

        public BertPredictions Predict(BertInput encodedInput)
        {
            return _predictionEngine.Predict(encodedInput);
        }
    }
}

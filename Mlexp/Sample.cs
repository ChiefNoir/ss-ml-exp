﻿using Microsoft.ML;
using Microsoft.ML.Data;

namespace Mlexp
{
    internal class Sample
    {
        public class HouseData
        {
            public float Size { get; set; }
            public float Price { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }

        public void DoWork()
        {
            MLContext mlContext = new MLContext();

            // 1. Load Data
            HouseData[] houseData =
            {
                new () { Size = 1.1F, Price = 1.2F },
                new () { Size = 1.9F, Price = 2.3F },
                new () { Size = 2.8F, Price = 3.0F },
                new () { Size = 3.4F, Price = 3.7F }
            };
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            // 2. Create pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" })
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            // 3. Train model
            var model = pipeline.Fit(trainingData);

            {
                // 4. Make a prediction and evaluate
                var size = new HouseData() { Size = 2.5F };
                var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

                Console.WriteLine($"Predicted price for size: {size.Size}; Price: {price.Price}");
            }
        }
    }
}

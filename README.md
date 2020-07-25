# Solar-Radiation-Prediction
Course project to predict solar radiation data using and RBF kernel SVM comparing its performance with a Neural Network.
It is an implementation of [Short Term Solar Power Prediction](https://www.sciencedirect.com/science/article/pii/S0960148112006465?casa_token=tsiGWFvcT3EAAAAA:1bv8HkGWANcir5-c9h0m37TsrzZCytBATQq72wvfbbSOQwpZdgm4kGQoQPthAequ8sickOGsBg)

# Description
The project takes current radiation data and weather conditions to predict the radiation for a particlar window (has to be decided beforehand). Later experimentation can involve using ensemble models to predict for multiple windows and to increase the performance.

## Dataset
We acquired the last 15 years hourly Solar Radiation Data from the [NSRDB Database](https://nsrdb.nrel.gov/). All the data used here is that for Mumbai from 2000-2015.

## Preprocessing
![Radiation 1D plot](https://github.com/ShaunZac/Solar-Radiation-Prediction/blob/master/plots/Hourly%20Radiation%20plot.png) ![Radiation 2D plot](https://github.com/ShaunZac/Solar-Radiation-Prediction/blob/master/plots/2D%20Solar%20Radiation.png)

The plots above show the two visualizations used for the dataset, on the left is a one-dimensional plot of the radiation, and on the right, there is a heatmap. The 1D plot is useful for seeing trends in seasons and the 2D plot is helpful for seeing trends in daily radiation values.

## Results
The results for Mean Average Error (MAE) and Mean Average Percentage Error have been plotted below for different prediction windows.
![MAE](https://github.com/ShaunZac/Solar-Radiation-Prediction/blob/master/plots/comparison%20plot%20MAE.png) ![MAPE](https://github.com/ShaunZac/Solar-Radiation-Prediction/blob/master/plots/comparison%20plot%20MAPE.png)

## Trends with Data Size
We experimented by changing the length of the dataset to see how it performs. The results of those experiments are plotted.
![MAE-MAPE](https://github.com/ShaunZac/Solar-Radiation-Prediction/blob/master/plots/MAE_MAPE%20(1%20hour).png)

## Conclusion
We tested the SVM model against the other model and found it to be superior in all prediction windows.

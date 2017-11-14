# aqi-forecasting
Created by wofmanaf， the main model we use is resnet!

### Requirements
* Keras 2.0.6
* Python 3.6
* Numpy 
* Pandas 

## Dateset
2015-2016 weather indicators and pollution indicators from Putuo District
Meteorological Bureau in Shanghai

example:
no,time,SO2,NO2,P10,P2.5,O3,CO,Humi-R,W-Direc,W-Speed,A-Pres,A-Temp
260,2015/1/1 12:00,0.022,0.02,0.064,0.025,0.068,0.401,27.778,245.638,2.009,1029.731,5.285
260,2015/1/1 13:00,0.022,0.021,0.064,0.023,0.07,0.412,27.488,216.073,1.838,1029.529,5.443

## Usage
1. use pre/pretreat.py to pretreat data 
2. use tools/model.py to test aqi_forecasting

We will provide document about aqi_forecasting soon!

Enjoying coding!

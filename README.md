# video morphing

1. Manually label corresponding points for some key frames
![tool](https://user-images.githubusercontent.com/5975007/90946854-ab159600-e3f6-11ea-9507-9217f9343a17.png)

2. Automatically generate corresponding points for all frames using optical flow
![of](https://user-images.githubusercontent.com/5975007/90946865-c7b1ce00-e3f6-11ea-9740-04b83887b878.png)

3. Spatiotemporal alignment: find a time interval that the correspoint points are mostly aligned
<img width="754" alt="align" src="https://user-images.githubusercontent.com/5975007/90946887-09427900-e3f7-11ea-8e9e-3fa5cccd22a6.png">

4. Image morphing for each frame in the interval

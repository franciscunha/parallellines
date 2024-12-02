# tinyrenderer

![](images/african_head.jpg)
*Image rendered with this rasterizer.*

Software rasterizer made while following the course https://github.com/ssloy/tinyrenderer.

The intent behind this project was to brush up on my C++ skills and get a fuller understanding of how graphics programming works on a lower level -- and I'm pretty happy with the results!

You should be able to write any shader you want by defining a class that inherits from `IStruct`, then render a model using the `renderer` API.

Next steps: Porting to the GPU with CUDA.

## Authorship

`main.cpp`, `line_renderer.*` and `renderer.*` were written completely by me. 

`geometry.*` and `model.*` are forked from an initial version by [ssloy](https://github.com/ssloy/) with many improvements to better fit my version of the rasterizer. `tgaimage.*` is fully written by [ssloy](https://github.com/ssloy/) for the course. 

## Resources used
- https://github.com/ssloy/tinyrenderer
- https://jtsorlinis.github.io/rendering-tutorial/
- https://www.songho.ca/opengl/gl_camera.html
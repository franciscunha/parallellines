#include <cmath>
#include <utility>
#include <iostream>
#include <algorithm>

#include "../include/tgaimage.hpp"
#include "../include/model.hpp"
#include "../include/line_renderer.hpp"

#include "../include/renderer.cuh"
#include "../include/shaders.cuh"


void renderer_demo() 
{
	TGAImage output(1024, 1024, TGAImage::RGB);

	Model model("models/african_head/african_head.obj");
	model.load_texture("models/african_head/african_head_diffuse.tga", TextureType::DIFFUSE);
	model.load_texture("models/african_head/african_head_spec.tga", TextureType::SPECULAR);
	model.load_texture("models/african_head/african_head_nm.tga", TextureType::NORMAL_MAP);

	renderer::render(output, model, PhongShader());

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("images/out.tga");
}

void lines_demo()
{
	TGAImage output(1024, 1024, TGAImage::RGB);
	Model model("models/african_head/african_head.obj");
	
	line_renderer::wireframe(model, output, TGAColor(255, 255, 255, 255));

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("images/out.tga");
}

int main(int argc, char **argv)
{
	renderer_demo();

	return 0;
}

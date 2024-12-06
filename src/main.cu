#include <cmath>
#include <utility>
#include <iostream>
#include <algorithm>

#include "../include/tgaimage.cuh"
#include "../include/model.cuh"
#include "../include/line_renderer.cuh"

#include "../include/renderer.cuh"
#include "../include/shaders.cuh"


void renderer_demo() 
{
	TGAImage output(256, 256, TGAImage::RGB);

	Model model("models/african_head/african_head.obj");
	model.load_texture("models/african_head/african_head_diffuse.tga", TextureType::DIFFUSE);
	model.load_texture("models/african_head/african_head_spec.tga", TextureType::SPECULAR);
	model.load_texture("models/african_head/african_head_nm.tga", TextureType::NORMAL_MAP);

	PhongShader shader = PhongShader();
	renderer::render(output, model, shader, sizeof(PhongShader));

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("images/out.tga");
}

void lines_demo()
{
	TGAImage output(256, 256, TGAImage::RGB);
	Model model("models/african_head/african_head.obj");
	
	line_renderer::wireframe(&model, &output, TGAColor(255, 255, 255, 255));

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("images/out.tga");
}

int main(int argc, char **argv)
{
	lines_demo();

	return 0;
}

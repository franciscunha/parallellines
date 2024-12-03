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

	PhongShader shader;

	shader.m_projection = renderer::projection(3);
	shader.m_view = renderer::loot_at(Vec3f(0.25f, 0.25f, 1));

	Matrix4 uniform_PVM = shader.m_projection * shader.m_view;
	if (!uniform_PVM.transpose().inverse(shader.uniform_PVM_it))
	{
		shader.uniform_PVM_it = Matrix4::identity();
	}
	shader.uniform_l = uniform_PVM.mult(shader.light_dir, false).normalize() * -1;

	renderer::render(output, model, shader);

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
